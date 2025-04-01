import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from miner.logic.training_worker import TrainingWorker


load_dotenv()


T = TypeVar("T", bound=BaseModel)


@dataclass
class WorkerConfig:
    trainer: TrainingWorker


@lru_cache
def factory_worker_config() -> WorkerConfig:
    # Get number of concurrent jobs from environment variable or default to 1
    max_concurrent_jobs = int(os.getenv("MINER_MAX_CONCURRENT_JOBS", "1"))
    
    return WorkerConfig(
        trainer=TrainingWorker(max_concurrent_jobs=max_concurrent_jobs),
    )