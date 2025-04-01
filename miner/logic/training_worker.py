import queue
import threading
import time
import traceback
from typing import Dict, List, Optional, Set
from uuid import UUID

import docker
import numpy as np
from fiber.logging_utils import get_logger

from core.models.utility_models import DiffusionJob
from core.models.utility_models import Job
from core.models.utility_models import JobStatus
from core.models.utility_models import TextJob
from miner.logic.job_handler import start_tuning_container
from miner.logic.job_handler import start_tuning_container_diffusion


logger = get_logger(__name__)


class PriorityJobQueue:
    """Priority queue for jobs based on model family expertise and historical performance."""
    
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.model_family_stats: Dict[str, Dict] = {}
        self.running_jobs: Dict[str, Dict] = {}
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        
    def get_model_family(self, model_name: str) -> str:
        """Extract model family from model name."""
        model_lower = model_name.lower()
        if "llama" in model_lower:
            return "llama"
        elif "mistral" in model_lower:
            return "mistral"
        elif "phi" in model_lower:
            return "phi"
        elif "stable-diffusion" in model_lower:
            return "stable-diffusion"
        elif "sdxl" in model_lower:
            return "sdxl"
        else:
            return "other"
            
    def update_model_family_stats(self, job_id: str, model: str, success: bool, loss: Optional[float] = None):
        """Update statistics for model families based on job results."""
        family = self.get_model_family(model)
        
        with self.lock:
            if family not in self.model_family_stats:
                self.model_family_stats[family] = {
                    "total_jobs": 0,
                    "success_jobs": 0,
                    "avg_loss": None,
                    "losses": []
                }
                
            stats = self.model_family_stats[family]
            stats["total_jobs"] += 1
            
            if success:
                stats["success_jobs"] += 1
                
            if loss is not None:
                stats["losses"].append(loss)
                stats["avg_loss"] = np.mean(stats["losses"])
                
            logger.info(f"Updated stats for {family}: {stats}")
        
    def put(self, job: Job):
        """Add job to queue with priority based on model family expertise."""
        family = self.get_model_family(job.model)
        priority = 1  # Default priority
        
        with self.lock:
            if family in self.model_family_stats:
                stats = self.model_family_stats[family]
                success_rate = stats["success_jobs"] / max(1, stats["total_jobs"])
                
                # Lower priority number = higher actual priority
                if success_rate > 0.9:
                    priority = 0  # High priority for families we're good at
                elif success_rate < 0.5:
                    priority = 2  # Low priority for families we struggle with
                    
            self.queue.put((priority, job))
            logger.info(f"Added job {job.job_id} to queue with priority {priority} (model family: {family})")
        
    def get(self, timeout=1):
        """Get highest priority job from queue with timeout to prevent blocking."""
        try:
            if self.queue.empty():
                return None
            
            # Get with timeout to prevent indefinite blocking
            _, job = self.queue.get(timeout=timeout)
            
            # Track start time for this job
            with self.lock:
                self.running_jobs[job.job_id] = {
                    "start_time": time.time(),
                    "model": job.model
                }
            
            return job
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting job from queue: {e}")
            return None
        
    def task_done(self, job_id: str, success: bool, metrics: Optional[Dict] = None):
        """Mark task as done and update stats."""
        loss = None
        if metrics and "loss" in metrics:
            loss = metrics.get("loss")
            
        with self.lock:
            if job_id in self.running_jobs:
                job_info = self.running_jobs[job_id]
                duration = time.time() - job_info["start_time"]
                
                # Update statistics for this model family
                self.update_model_family_stats(job_id, job_info["model"], success, loss)
                
                logger.info(f"Job {job_id} completed in {duration:.2f}s (success: {success})")
                del self.running_jobs[job_id]
                
            try:
                self.queue.task_done()
            except ValueError:
                # Ignore if the queue is empty
                pass


class TrainingWorker:
    def __init__(self, max_concurrent_jobs=1):
        logger.info("=" * 80)
        logger.info("STARTING AN OPTIMIZED TRAINING WORKER")
        logger.info("=" * 80)

        self.job_queue = PriorityJobQueue()
        self.job_store: dict[str, Job] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs = 0
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.job_semaphore = threading.Semaphore(max_concurrent_jobs)
        self.threads: List[threading.Thread] = []
        self.shutdown_flag = threading.Event()
        self.docker_client = docker.from_env()
        
        # Start worker threads
        for i in range(max_concurrent_jobs):
            thread = threading.Thread(target=self._worker, daemon=True, name=f"worker-{i}")
            thread.start()
            self.threads.append(thread)
            
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True)
        self.monitor_thread.start()
        
        # Start watchdog thread
        self.watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self.watchdog_thread.start()
        
        # Track failed models to avoid accepting similar jobs
        self.failed_models: Set[str] = set()
        
    def _watchdog(self):
        """Watchdog thread to ensure worker threads are running and responsive."""
        while not self.shutdown_flag.is_set():
            try:
                # Check if any jobs are stuck
                with self.lock:
                    for job_id, job in self.job_store.items():
                        # If a job has been in PROCESSING state for more than 1 hour, something might be wrong
                        if (job.status == JobStatus.PROCESSING and 
                            job_id in self.job_queue.running_jobs and
                            time.time() - self.job_queue.running_jobs[job_id]["start_time"] > 3600):
                            logger.warning(f"Job {job_id} appears to be stuck in PROCESSING state for >1 hour")
                            
                            # Attempt to fix the situation
                            try:
                                # Reset active jobs count if needed
                                if self.active_jobs > 0:
                                    logger.warning(f"Resetting active_jobs count from {self.active_jobs} to 0")
                                    self.active_jobs = 0
                                    
                                # Release semaphore if needed
                                self.job_semaphore.release()
                                logger.warning(f"Released job semaphore to unblock worker threads")
                            except Exception as e:
                                logger.error(f"Error resetting worker state: {e}")
                
                # Sleep for 5 minutes before checking again
                time.sleep(300)
            except Exception as e:
                logger.error(f"Error in watchdog thread: {e}")
                time.sleep(60)  # Sleep for a minute if there's an error
        
    def _worker(self):
        """Worker thread that processes jobs from the queue."""
        logger.info(f"Worker thread {threading.current_thread().name} started")
        
        while not self.shutdown_flag.is_set():
            try:
                # Wait for semaphore (job slot available)
                acquired = False
                try:
                    acquired = self.job_semaphore.acquire(timeout=5)  # Wait with timeout
                    if not acquired:
                        # Timeout waiting for semaphore, continue loop
                        continue
                except Exception as e:
                    logger.error(f"Error acquiring semaphore: {e}")
                    time.sleep(1)
                    continue
                
                # Get job from queue
                job = self.job_queue.get(timeout=5)  # Use timeout to prevent blocking
                if job is None:
                    if acquired:
                        self.job_semaphore.release()
                    time.sleep(1)  # No job available, sleep briefly
                    continue
                    
                # Update job status and active job count
                with self.lock:
                    self.active_jobs += 1
                    job.status = JobStatus.PROCESSING
                
                logger.info(f"Processing job {job.job_id} ({self.active_jobs}/{self.max_concurrent_jobs} active jobs)")
                metrics = None
                success = False
                
                try:
                    # Process job based on type
                    if isinstance(job, TextJob):
                        metrics = start_tuning_container(job)
                    elif isinstance(job, DiffusionJob):
                        metrics = start_tuning_container_diffusion(job)
                    else:
                        logger.error(f"Unknown job type: {type(job)}")
                        raise ValueError(f"Unknown job type: {type(job)}")
                        
                    job.status = JobStatus.COMPLETED
                    logger.info(f"Successfully completed job {job.job_id} for model {job.model}")
                    success = True
                    
                except Exception as e:
                    error_trace = traceback.format_exc()
                    logger.error(f"Error processing job {job.job_id}: {str(e)}")
                    logger.error(f"Error trace: {error_trace}")
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    
                    # Add to failed models set
                    model_family = self.job_queue.get_model_family(job.model)
                    with self.lock:
                        self.failed_models.add(model_family)
                    logger.warning(f"Added {model_family} to failed models set")
                    
                    success = False
                    
                finally:
                    # Update job queue stats
                    self.job_queue.task_done(job.job_id, success, metrics)
                    
                    # Release resources
                    with self.lock:
                        self.active_jobs = max(0, self.active_jobs - 1)  # Prevent negative counts
                    
                    # Release semaphore
                    if acquired:
                        self.job_semaphore.release()
                
            except Exception as e:
                logger.error(f"Unhandled error in worker thread: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1)  # Sleep to avoid tight loop in case of persistent error
                
                # Try to release semaphore if we had acquired it
                try:
                    if 'acquired' in locals() and acquired:
                        self.job_semaphore.release()
                except Exception:
                    pass

    def _monitor_jobs(self):
        """Monitor thread to log system status and job queue information."""
        try_restart_count = 0
        
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(60)  # Update every minute
                
                # Current queue size is approximate since we don't lock the queue
                approximate_queue_size = self.job_queue.queue.qsize()
                
                # Log current status
                logger.info(f"-- Job Queue Status --")
                logger.info(f"Active jobs: {self.active_jobs}/{self.max_concurrent_jobs}")
                logger.info(f"Queue size: approximately {approximate_queue_size} jobs")
                logger.info(f"Model family stats: {self.job_queue.model_family_stats}")
                logger.info(f"Failed model families: {self.failed_models}")
                logger.info(f"----------------------")
                
                # Check if we need to restart processing
                if (approximate_queue_size > 0 and self.active_jobs == 0 and 
                    not self.shutdown_flag.is_set()):
                    try_restart_count += 1
                    
                    if try_restart_count >= 5:  # After 5 minutes of having jobs but no activity
                        logger.warning("Detected jobs in queue but no active jobs for 5 minutes")
                        logger.warning("Attempting to reset worker state...")
                        
                        # Release all semaphores to unblock workers
                        for _ in range(self.max_concurrent_jobs):
                            try:
                                self.job_semaphore.release()
                            except Exception:
                                pass
                            
                        try_restart_count = 0  # Reset counter
                else:
                    try_restart_count = 0  # Reset if conditions change
                
            except Exception as e:
                logger.error(f"Error in monitor thread: {e}")

    def enqueue_job(self, job: Job):
        with self.lock:
            job.status = JobStatus.QUEUED
            self.job_store[job.job_id] = job
            
        self.job_queue.put(job)
        logger.info(f"Enqueued job {job.job_id}")

    def get_status(self, job_id: UUID) -> JobStatus:
        job_id_str = str(job_id)
        with self.lock:
            job = self.job_store.get(job_id_str)
            return job.status if job else JobStatus.NOT_FOUND

    def can_accept_model(self, model: str) -> bool:
        """Check if we should accept a job for this model based on past performance."""
        model_family = self.job_queue.get_model_family(model)
        
        # Check if model family is in failed models set
        if model_family in self.failed_models:
            logger.info(f"Rejecting job for model family {model_family} due to past failures")
            return False
            
        # Check if we have stats for this model family
        with self.lock:
            if model_family in self.job_queue.model_family_stats:
                stats = self.job_queue.model_family_stats[model_family]
                success_rate = stats["success_jobs"] / max(1, stats["total_jobs"])
                
                # Only accept model families with >50% success rate after multiple attempts
                if success_rate < 0.5 and stats["total_jobs"] >= 2:
                    logger.info(f"Rejecting job for model family {model_family} due to low success rate ({success_rate:.2f})")
                    return False
                
        # For new model families we haven't seen before, accept them
        return True

    def shutdown(self):
        logger.info("Shutting down training worker...")
        self.shutdown_flag.set()
        
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
            
        self.monitor_thread.join(timeout=5)
        self.watchdog_thread.join(timeout=5)
        
        # Close docker client
        try:
            self.docker_client.close()
        except Exception as e:
            logger.error(f"Error closing docker client: {e}")
        
        logger.info("Training worker shutdown complete")