import queue
import threading
import time
from typing import Dict, List, Optional
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
        
    def get(self):
        """Get highest priority job from queue."""
        if self.queue.empty():
            return None
        _, job = self.queue.get()
        # Track start time for this job
        self.running_jobs[job.job_id] = {
            "start_time": time.time(),
            "model": job.model
        }
        return job
        
    def task_done(self, job_id: str, success: bool):
        """Mark task as done and update stats."""
        if job_id in self.running_jobs:
            job_info = self.running_jobs[job_id]
            duration = time.time() - job_info["start_time"]
            
            # Update statistics for this model family
            self.update_model_family_stats(job_id, job_info["model"], success)
            
            logger.info(f"Job {job_id} completed in {duration:.2f}s (success: {success})")
            del self.running_jobs[job_id]
            
        self.queue.task_done()


class TrainingWorker:
    def __init__(self, max_concurrent_jobs=1):
        logger.info("=" * 80)
        logger.info("STARTING AN OPTIMIZED TRAINING WORKER")
        logger.info("=" * 80)

        self.job_queue = PriorityJobQueue()
        self.job_store: dict[str, Job] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs = 0
        self.job_semaphore = threading.Semaphore(max_concurrent_jobs)
        self.threads: List[threading.Thread] = []
        self.docker_client = docker.from_env()
        
        # Start worker threads
        for i in range(max_concurrent_jobs):
            thread = threading.Thread(target=self._worker, daemon=True, name=f"worker-{i}")
            thread.start()
            self.threads.append(thread)
            
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True)
        self.monitor_thread.start()
        
        # Track failed models to avoid accepting similar jobs
        self.failed_models = set()
        
    def _worker(self):
        """Worker thread that processes jobs from the queue."""
        while True:
            # Wait for semaphore (job slot available)
            self.job_semaphore.acquire()
            
            # Get job from queue
            job = self.job_queue.get()
            if job is None:
                self.job_semaphore.release()
                break
                
            self.active_jobs += 1
            try:
                logger.info(f"Processing job {job.job_id} ({self.active_jobs}/{self.max_concurrent_jobs} active jobs)")
                
                # Process job based on type
                if isinstance(job, TextJob):
                    start_tuning_container(job)
                elif isinstance(job, DiffusionJob):
                    start_tuning_container_diffusion(job)
                    
                job.status = JobStatus.COMPLETED
                self.job_queue.task_done(job.job_id, True)
                
                # Add model to successful models list
                logger.info(f"Successfully completed job {job.job_id} for model {job.model}")
                
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                self.job_queue.task_done(job.job_id, False)
                
                # Add to failed models set
                model_family = self.job_queue.get_model_family(job.model)
                self.failed_models.add(model_family)
                logger.warning(f"Added {model_family} to failed models set")
                
            finally:
                self.active_jobs -= 1
                self.job_semaphore.release()

    def _monitor_jobs(self):
        """Monitor thread to log system status and job queue information."""
        while True:
            time.sleep(60)  # Update every minute
            
            # Log current status
            logger.info(f"-- Job Queue Status --")
            logger.info(f"Active jobs: {self.active_jobs}/{self.max_concurrent_jobs}")
            logger.info(f"Queue size: approximately {self.job_queue.queue.qsize()} jobs")
            logger.info(f"Model family stats: {self.job_queue.model_family_stats}")
            logger.info(f"Failed model families: {self.failed_models}")
            logger.info(f"----------------------")

    def enqueue_job(self, job: Job):
        self.job_queue.put(job)
        self.job_store[job.job_id] = job
        logger.info(f"Enqueued job {job.job_id}")

    def get_status(self, job_id: UUID) -> JobStatus:
        job = self.job_store.get(str(job_id))
        return job.status if job else JobStatus.NOT_FOUND

    def can_accept_model(self, model: str) -> bool:
        """Check if we should accept a job for this model based on past performance."""
        model_family = self.job_queue.get_model_family(model)
        
        # Check if model family is in failed models set
        if model_family in self.failed_models:
            logger.info(f"Rejecting job for model family {model_family} due to past failures")
            return False
            
        # Check if we have stats for this model family
        if model_family in self.job_queue.model_family_stats:
            stats = self.job_queue.model_family_stats[model_family]
            success_rate = stats["success_jobs"] / max(1, stats["total_jobs"])
            
            # Only accept model families with >50% success rate
            if success_rate < 0.5 and stats["total_jobs"] >= 2:
                logger.info(f"Rejecting job for model family {model_family} due to low success rate ({success_rate:.2f})")
                return False
                
        # For new model families we haven't seen before, accept them
        return True

    def shutdown(self):
        # Signal all worker threads to exit
        for _ in range(len(self.threads)):
            self.job_queue.put(None)
            
        # Wait for all threads to finish
        for thread in self.threads:
            thread.join()
            
        self.monitor_thread.join()
        self.docker_client.close()