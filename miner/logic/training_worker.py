import queue
import threading
import time
import traceback
import json
import os
import sys
from datetime import datetime
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

# Create a directory for diagnostic logs if it doesn't exist
DIAGNOSTIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
DIAGNOSTIC_LOG = os.path.join(DIAGNOSTIC_DIR, "miner_diagnostics.log")

def log_diagnostic(message, level="INFO", include_trace=False):
    """Log diagnostic information to a separate file for easy retrieval."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(DIAGNOSTIC_LOG, "a") as f:
            log_entry = f"{timestamp} | {level} | {message}"
            if include_trace and level in ["ERROR", "WARNING"]:
                log_entry += f"\n{traceback.format_exc()}"
            f.write(log_entry + "\n")
            
        # Always log to regular logger as well
        if level == "INFO":
            logger.info(f"DIAGNOSTIC: {message}")
        elif level == "WARNING":
            logger.warning(f"DIAGNOSTIC: {message}")
        elif level == "ERROR":
            logger.error(f"DIAGNOSTIC: {message}")
        elif level == "DEBUG":
            logger.debug(f"DIAGNOSTIC: {message}")
    except Exception as e:
        logger.error(f"Failed to write diagnostic log: {e}")


class PriorityJobQueue:
    """Priority queue for jobs based on model family expertise and historical performance."""
    
    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.model_family_stats: Dict[str, Dict] = {}
        self.running_jobs: Dict[str, Dict] = {}
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        log_diagnostic(f"Initialized PriorityJobQueue", "INFO")
        
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
            
            stats_msg = f"Updated stats for {family}: {stats}"
            log_diagnostic(stats_msg, "INFO")
            logger.info(stats_msg)
        
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
            log_msg = f"Added job {job.job_id} to queue with priority {priority} (model family: {family})"
            log_diagnostic(log_msg, "INFO")
            logger.info(log_msg)
        
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
            
            log_diagnostic(f"Retrieved job {job.job_id} from queue (model: {job.model})", "INFO")
            return job
        except queue.Empty:
            return None
        except Exception as e:
            error_msg = f"Error getting job from queue: {e}"
            log_diagnostic(error_msg, "ERROR", include_trace=True)
            logger.error(error_msg)
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
                
                status = "successfully" if success else "unsuccessfully"
                metrics_str = f" with metrics {metrics}" if metrics else ""
                log_msg = f"Job {job_id} completed {status} in {duration:.2f}s{metrics_str}"
                log_diagnostic(log_msg, "INFO")
                logger.info(log_msg)
                
                del self.running_jobs[job_id]
                
            try:
                self.queue.task_done()
            except ValueError:
                # Ignore if the queue is empty
                pass


class TrainingWorker:
    def __init__(self, max_concurrent_jobs=1):
        worker_start_msg = "=" * 80 + "\nSTARTING AN OPTIMIZED TRAINING WORKER\n" + "=" * 80
        log_diagnostic(worker_start_msg, "INFO")
        logger.info(worker_start_msg)

        self.job_queue = PriorityJobQueue()
        self.job_store: dict[str, Job] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs = 0
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.job_semaphore = threading.Semaphore(max_concurrent_jobs)
        self.threads: List[threading.Thread] = []
        self.shutdown_flag = threading.Event()
        self.docker_client = docker.from_env()
        
        # Configuration logging
        config_info = {
            "max_concurrent_jobs": max_concurrent_jobs,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "miner_version": "1.0.1-custom"  # Update this with your version
        }
        log_diagnostic(f"Miner configuration: {json.dumps(config_info)}", "INFO")
        
        # Start worker threads
        for i in range(max_concurrent_jobs):
            thread = threading.Thread(target=self._worker, daemon=True, name=f"worker-{i}")
            thread.start()
            self.threads.append(thread)
            log_diagnostic(f"Started worker thread: {thread.name}", "INFO")
            
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True, name="monitor")
        self.monitor_thread.start()
        log_diagnostic("Started monitor thread", "INFO")
        
        # Start watchdog thread
        self.watchdog_thread = threading.Thread(target=self._watchdog, daemon=True, name="watchdog")
        self.watchdog_thread.start()
        log_diagnostic("Started watchdog thread", "INFO")
        
        # Start health check thread
        self.health_check_thread = threading.Thread(target=self._health_checker, daemon=True, name="health-check")
        self.health_check_thread.start()
        log_diagnostic("Started health check thread", "INFO")
        
        # Track failed models to avoid accepting similar jobs
        self.failed_models: Set[str] = set()
        
    def _health_checker(self):
        """Thread to periodically check system health and log diagnostics."""
        while not self.shutdown_flag.is_set():
            try:
                # Check Docker service
                try:
                    docker_info = self.docker_client.info()
                    log_diagnostic(f"Docker status: running, containers: {docker_info.get('ContainersRunning', 'unknown')}", "INFO")
                except Exception as e:
                    log_diagnostic(f"Docker service check failed: {e}", "ERROR")
                
                # Check GPU status if nvidia-smi is available
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], 
                                           capture_output=True, text=True, check=True)
                    log_diagnostic(f"GPU status: {result.stdout.strip()}", "INFO")
                except (subprocess.SubprocessError, FileNotFoundError) as e:
                    log_diagnostic(f"GPU status check failed or not available: {e}", "DEBUG")
                
                # Check thread status
                thread_status = {
                    "active_threads": threading.active_count(),
                    "worker_threads_alive": sum(1 for t in self.threads if t.is_alive()),
                    "monitor_alive": self.monitor_thread.is_alive(),
                    "watchdog_alive": self.watchdog_thread.is_alive()
                }
                log_diagnostic(f"Thread status: {json.dumps(thread_status)}", "INFO")
                
                # Check queue and job status
                with self.lock:
                    job_stats = {
                        "active_jobs": self.active_jobs,
                        "queue_size": self.job_queue.queue.qsize(),
                        "job_store_size": len(self.job_store),
                        "running_jobs": len(self.job_queue.running_jobs),
                        "model_families_stats_count": len(self.job_queue.model_family_stats),
                        "failed_models_count": len(self.failed_models)
                    }
                log_diagnostic(f"Job stats: {json.dumps(job_stats)}", "INFO")
                
                # Sleep for 15 minutes before next check
                time.sleep(900)
                
            except Exception as e:
                log_diagnostic(f"Error in health check thread: {e}", "ERROR", include_trace=True)
                time.sleep(60)  # Sleep for a minute if there's an error
        
    def _watchdog(self):
        """Watchdog thread to ensure worker threads are running and responsive."""
        log_diagnostic("Watchdog thread started", "INFO")
        
        while not self.shutdown_flag.is_set():
            try:
                # Check if any jobs are stuck
                with self.lock:
                    for job_id, job in self.job_store.items():
                        # If a job has been in PROCESSING state for more than 1 hour, something might be wrong
                        if (job.status == JobStatus.PROCESSING and 
                            job_id in self.job_queue.running_jobs and
                            time.time() - self.job_queue.running_jobs[job_id]["start_time"] > 3600):
                            
                            stuck_msg = f"Job {job_id} appears to be stuck in PROCESSING state for >1 hour"
                            log_diagnostic(stuck_msg, "WARNING")
                            logger.warning(stuck_msg)
                            
                            # Attempt to fix the situation
                            try:
                                # Reset active jobs count if needed
                                if self.active_jobs > 0:
                                    reset_msg = f"Resetting active_jobs count from {self.active_jobs} to 0"
                                    log_diagnostic(reset_msg, "WARNING")
                                    logger.warning(reset_msg)
                                    self.active_jobs = 0
                                    
                                # Release semaphore if needed
                                self.job_semaphore.release()
                                log_diagnostic("Released job semaphore to unblock worker threads", "WARNING")
                                logger.warning("Released job semaphore to unblock worker threads")
                            except Exception as e:
                                error_msg = f"Error resetting worker state: {e}"
                                log_diagnostic(error_msg, "ERROR", include_trace=True)
                                logger.error(error_msg)
                
                # Check if worker threads are alive, restart if needed
                for i, thread in enumerate(self.threads):
                    if not thread.is_alive():
                        thread_error = f"Worker thread {thread.name} is not alive, starting a new one"
                        log_diagnostic(thread_error, "WARNING")
                        logger.warning(thread_error)
                        
                        # Create and start a new thread
                        new_thread = threading.Thread(target=self._worker, daemon=True, name=f"worker-{i}-restarted")
                        new_thread.start()
                        self.threads[i] = new_thread
                        log_diagnostic(f"Started replacement worker thread: {new_thread.name}", "INFO")
                
                # Sleep for 5 minutes before checking again
                time.sleep(300)
            except Exception as e:
                error_msg = f"Error in watchdog thread: {e}"
                log_diagnostic(error_msg, "ERROR", include_trace=True)
                logger.error(error_msg)
                time.sleep(60)  # Sleep for a minute if there's an error
        
    def _worker(self):
        """Worker thread that processes jobs from the queue."""
        thread_name = threading.current_thread().name
        log_diagnostic(f"Worker thread {thread_name} started", "INFO")
        logger.info(f"Worker thread {thread_name} started")
        
        while not self.shutdown_flag.is_set():
            try:
                # Wait for semaphore (job slot available)
                acquired = False
                try:
                    log_diagnostic(f"Worker {thread_name} waiting for semaphore", "DEBUG")
                    acquired = self.job_semaphore.acquire(timeout=5)  # Wait with timeout
                    if not acquired:
                        # Timeout waiting for semaphore, continue loop
                        continue
                    log_diagnostic(f"Worker {thread_name} acquired semaphore", "DEBUG")
                except Exception as e:
                    error_msg = f"Worker {thread_name} error acquiring semaphore: {e}"
                    log_diagnostic(error_msg, "ERROR", include_trace=True)
                    logger.error(error_msg)
                    time.sleep(1)
                    continue
                
                # Get job from queue
                log_diagnostic(f"Worker {thread_name} attempting to get job from queue", "DEBUG")
                job = self.job_queue.get(timeout=5)  # Use timeout to prevent blocking
                if job is None:
                    if acquired:
                        log_diagnostic(f"Worker {thread_name} releasing semaphore (no job available)", "DEBUG")
                        self.job_semaphore.release()
                    time.sleep(1)  # No job available, sleep briefly
                    continue
                
                log_diagnostic(f"Worker {thread_name} got job {job.job_id} from queue", "INFO")
                    
                # Update job status and active job count
                with self.lock:
                    self.active_jobs += 1
                    job.status = JobStatus.PROCESSING
                
                job_start_msg = f"Processing job {job.job_id} (model: {job.model}, {self.active_jobs}/{self.max_concurrent_jobs} active jobs)"
                log_diagnostic(job_start_msg, "INFO")
                logger.info(job_start_msg)
                
                metrics = None
                success = False
                
                try:
                    # Process job based on type
                    if isinstance(job, TextJob):
                        log_diagnostic(f"Starting text model tuning for job {job.job_id}", "INFO")
                        metrics = start_tuning_container(job)
                    elif isinstance(job, DiffusionJob):
                        log_diagnostic(f"Starting diffusion model tuning for job {job.job_id}", "INFO")
                        metrics = start_tuning_container_diffusion(job)
                    else:
                        error_msg = f"Unknown job type: {type(job)}"
                        log_diagnostic(error_msg, "ERROR")
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                        
                    job.status = JobStatus.COMPLETED
                    success_msg = f"Successfully completed job {job.job_id} for model {job.model}"
                    if metrics:
                        success_msg += f" with metrics: {json.dumps(metrics)}"
                        
                    log_diagnostic(success_msg, "INFO")
                    logger.info(success_msg)
                    success = True
                    
                except Exception as e:
                    error_trace = traceback.format_exc()
                    error_msg = f"Error processing job {job.job_id}: {str(e)}"
                    log_diagnostic(error_msg, "ERROR", include_trace=True)
                    logger.error(error_msg)
                    logger.error(f"Error trace: {error_trace}")
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                    
                    # Add to failed models set
                    model_family = self.job_queue.get_model_family(job.model)
                    with self.lock:
                        self.failed_models.add(model_family)
                    
                    warning_msg = f"Added {model_family} to failed models set due to error in job {job.job_id}"
                    log_diagnostic(warning_msg, "WARNING") 
                    logger.warning(warning_msg)
                    
                    success = False
                    
                finally:
                    # Update job queue stats
                    self.job_queue.task_done(job.job_id, success, metrics)
                    
                    # Release resources
                    with self.lock:
                        self.active_jobs = max(0, self.active_jobs - 1)  # Prevent negative counts
                    
                    # Release semaphore
                    if acquired:
                        log_diagnostic(f"Worker {thread_name} releasing semaphore after job {job.job_id}", "DEBUG")
                        self.job_semaphore.release()
                
            except Exception as e:
                error_msg = f"Unhandled error in worker thread {thread_name}: {e}"
                log_diagnostic(error_msg, "ERROR", include_trace=True)
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                time.sleep(1)  # Sleep to avoid tight loop in case of persistent error
                
                # Try to release semaphore if we had acquired it
                try:
                    if 'acquired' in locals() and acquired:
                        log_diagnostic(f"Worker {thread_name} emergency semaphore release", "WARNING")
                        self.job_semaphore.release()
                except Exception:
                    pass

    def _monitor_jobs(self):
        """Monitor thread to log system status and job queue information."""
        log_diagnostic("Monitor thread started", "INFO")
        try_restart_count = 0
        
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(60)  # Update every minute
                
                # Current queue size is approximate since we don't lock the queue
                approximate_queue_size = self.job_queue.queue.qsize()
                
                # Log current status
                status_msg = f"-- Job Queue Status --\n"
                status_msg += f"Active jobs: {self.active_jobs}/{self.max_concurrent_jobs}\n"
                status_msg += f"Queue size: approximately {approximate_queue_size} jobs\n"
                status_msg += f"Model family stats: {json.dumps(self.job_queue.model_family_stats)}\n"
                status_msg += f"Failed model families: {list(self.failed_models)}\n"
                status_msg += f"Running jobs: {list(self.job_queue.running_jobs.keys())}\n"
                status_msg += f"----------------------"
                
                log_diagnostic(status_msg, "INFO")
                
                # Log status to regular logger in simplified form
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
                    log_diagnostic(f"Detected jobs in queue but no active jobs (restart attempt count: {try_restart_count})", "WARNING")
                    
                    if try_restart_count >= 5:  # After 5 minutes of having jobs but no activity
                        warning_msg = "Detected jobs in queue but no active jobs for 5 minutes. Attempting to reset worker state..."
                        log_diagnostic(warning_msg, "WARNING")
                        logger.warning(warning_msg)
                        
                        # Release all semaphores to unblock workers
                        for i in range(self.max_concurrent_jobs):
                            try:
                                self.job_semaphore.release()
                                log_diagnostic(f"Released semaphore #{i+1} to unblock workers", "INFO")
                            except Exception as e:
                                log_diagnostic(f"Failed to release semaphore #{i+1}: {e}", "ERROR")
                            
                        try_restart_count = 0  # Reset counter
                else:
                    try_restart_count = 0  # Reset if conditions change
                
            except Exception as e:
                error_msg = f"Error in monitor thread: {e}"
                log_diagnostic(error_msg, "ERROR", include_trace=True)
                logger.error(error_msg)

    def enqueue_job(self, job: Job):
        job_id = job.job_id
        log_diagnostic(f"Enqueueing job {job_id} (model: {job.model})", "INFO")
        
        with self.lock:
            job.status = JobStatus.QUEUED
            self.job_store[job_id] = job
            
        self.job_queue.put(job)
        logger.info(f"Enqueued job {job_id}")

    def get_status(self, job_id: UUID) -> JobStatus:
        job_id_str = str(job_id)
        with self.lock:
            job = self.job_store.get(job_id_str)
            status = job.status if job else JobStatus.NOT_FOUND
            
        log_diagnostic(f"Status request for job {job_id_str}: {status}", "INFO")
        return status

    def can_accept_model(self, model: str) -> bool:
        """Check if we should accept a job for this model based on past performance."""
        model_family = self.job_queue.get_model_family(model)
        
        # Check if model family is in failed models set
        if model_family in self.failed_models:
            log_msg = f"Rejecting job for model family {model_family} due to past failures"
            log_diagnostic(log_msg, "INFO")
            logger.info(log_msg)
            return False
            
        # Check if we have stats for this model family
        with self.lock:
            if model_family in self.job_queue.model_family_stats:
                stats = self.job_queue.model_family_stats[model_family]
                success_rate = stats["success_jobs"] / max(1, stats["total_jobs"])
                
                # Only accept model families with >50% success rate after multiple attempts
                if success_rate < 0.5 and stats["total_jobs"] >= 2:
                    log_msg = f"Rejecting job for model family {model_family} due to low success rate ({success_rate:.2f})"
                    log_diagnostic(log_msg, "INFO")
                    logger.info(log_msg)
                    return False
        
        log_diagnostic(f"Accepting job for model family {model_family}", "INFO")
        return True

    def shutdown(self):
        log_msg = "Shutting down training worker..."
        log_diagnostic(log_msg, "INFO")
        logger.info(log_msg)
        
        self.shutdown_flag.set()
        
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
            
        self.monitor_thread.join(timeout=5)
        self.watchdog_thread.join(timeout=5)
        self.health_check_thread.join(timeout=5)
        
        # Close docker client
        try:
            self.docker_client.close()
        except Exception as e:
            error_msg = f"Error closing docker client: {e}"
            log_diagnostic(error_msg, "ERROR")
            logger.error(error_msg)
        
        log_msg = "Training worker shutdown complete"
        log_diagnostic(log_msg, "INFO")
        logger.info(log_msg)