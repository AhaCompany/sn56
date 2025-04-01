import os
from datetime import datetime
from datetime import timedelta

import toml
import yaml
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import get_config
from fiber.miner.dependencies import verify_request
from pydantic import ValidationError

import core.constants as cst
from core.models.payload_models import MinerTaskOffer
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job_diffusion
from miner.logic.job_handler import create_job_text


logger = get_logger(__name__)

current_job_finish_time = None
# Track acceptable hours to complete for different model types
MAX_TEXT_TASK_HOURS = int(os.getenv("MAX_TEXT_TASK_HOURS", "12"))
MAX_IMAGE_TASK_HOURS = int(os.getenv("MAX_IMAGE_TASK_HOURS", "3"))
# Track acceptable model families
ACCEPTED_MODEL_FAMILIES = os.getenv("ACCEPTED_MODEL_FAMILIES", "llama,mistral,qwen,samoline").lower().split(",")
# Track queue capacity
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "5"))


async def tune_model_text(
    train_request: TrainRequestText,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")

    try:
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
                train_request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def tune_model_diffusion(
    train_request: TrainRequestImage,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")
    try:
        train_request.dataset_zip = await download_s3_file(
            train_request.dataset_zip, f"{cst.DIFFUSION_DATASET_DIR}/{train_request.task_id}.zip"
        )
        logger.info(train_request.dataset_zip)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_diffusion(
        job_id=str(train_request.task_id),
        dataset_zip=train_request.dataset_zip,
        model=train_request.model,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def get_latest_model_submission(task_id: str) -> str:
    try:
        # Try YAML config first (text tasks)
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
                return config_data.get("hub_model_id", None)
        else:
            # Try TOML config next (diffusion tasks)
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)
            with open(config_path, "r") as file:
                config_data = toml.load(file)
                return config_data.get("huggingface_repo_id", None)

    except FileNotFoundError as e:
        logger.error(f"No submission found for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest model submission: {str(e)}",
        )


async def task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info(f"Text task offer received: {request}")
        
        # Basic validation
        if request.task_type != TaskType.TEXTTASK:
            return MinerTaskResponse(message="This endpoint only accepts text tasks", accepted=False)

        # Check if we're already at capacity with the queue
        queue_size = worker_config.trainer.job_queue.queue.qsize()
        if queue_size >= MAX_QUEUE_SIZE:
            logger.info(f"Rejecting offer due to queue capacity ({queue_size}/{MAX_QUEUE_SIZE})")
            return MinerTaskResponse(
                message=f"Queue at capacity ({queue_size}/{MAX_QUEUE_SIZE})", accepted=False
            )

        # Extract model family and check if we accept it
        model_lower = request.model.lower()
        model_family = None
        for family in ACCEPTED_MODEL_FAMILIES:
            if family in model_lower:
                model_family = family
                break
                
        if model_family is None:
            logger.info(f"Rejecting offer for unsupported model family: {request.model}")
            return MinerTaskResponse(
                message=f"Only accepting these model families: {', '.join(ACCEPTED_MODEL_FAMILIES)}", 
                accepted=False
            )
            
        # Check if model has failed previously
        if not worker_config.trainer.can_accept_model(request.model):
            return MinerTaskResponse(
                message=f"This model family has failed in previous attempts", 
                accepted=False
            )

        # Check time constraints
        global current_job_finish_time
        current_time = datetime.now()
        
        # If no current job or current job is almost done, we can accept new ones
        if current_job_finish_time is None or current_time + timedelta(hours=1) > current_job_finish_time:
            # Check if job can be completed in our acceptable time frame
            if request.hours_to_complete <= MAX_TEXT_TASK_HOURS:
                logger.info(f"Accepting text task offer for model family {model_family}")
                return MinerTaskResponse(message=f"Yes. I can do {request.task_type} jobs", accepted=True)
            else:
                logger.info(f"Rejecting text task offer due to time constraints: {request.hours_to_complete}h > {MAX_TEXT_TASK_HOURS}h")
                return MinerTaskResponse(
                    message=f"I only accept jobs requiring {MAX_TEXT_TASK_HOURS}h or less (requested: {request.hours_to_complete}h)", 
                    accepted=False
                )
        else:
            # We're busy with current job
            return MinerTaskResponse(
                message=f"Currently busy with another job until {current_job_finish_time.isoformat()}",
                accepted=False,
            )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def task_offer_image(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info(f"Image task offer received: {request}")
        
        # Basic validation
        if request.task_type != TaskType.IMAGETASK:
            return MinerTaskResponse(message="This endpoint only accepts image tasks", accepted=False)

        # Check if we're already at capacity with the queue
        queue_size = worker_config.trainer.job_queue.queue.qsize()
        if queue_size >= MAX_QUEUE_SIZE:
            logger.info(f"Rejecting offer due to queue capacity ({queue_size}/{MAX_QUEUE_SIZE})")
            return MinerTaskResponse(
                message=f"Queue at capacity ({queue_size}/{MAX_QUEUE_SIZE})", accepted=False
            )
            
        # Check if diffusion model is in acceptable list - simplifying here, we accept all diffusion models
        # For production you'd want to filter like we do for text models
        if not worker_config.trainer.can_accept_model(request.model):
            return MinerTaskResponse(
                message=f"This model family has failed in previous attempts", 
                accepted=False
            )

        # Check time constraints
        global current_job_finish_time
        current_time = datetime.now()
        
        # If no current job or current job is almost done, we can accept new ones
        if current_job_finish_time is None or current_time + timedelta(hours=1) > current_job_finish_time:
            # Check if job can be completed in our acceptable time frame
            if request.hours_to_complete <= MAX_IMAGE_TASK_HOURS:
                logger.info("Accepting image task offer")
                return MinerTaskResponse(message="Yes. I can do image jobs", accepted=True)
            else:
                logger.info(f"Rejecting image task offer due to time constraints: {request.hours_to_complete}h > {MAX_IMAGE_TASK_HOURS}h")
                return MinerTaskResponse(
                    message=f"I only accept jobs requiring {MAX_IMAGE_TASK_HOURS}h or less (requested: {request.hours_to_complete}h)", 
                    accepted=False
                )
        else:
            # We're busy with current job
            return MinerTaskResponse(
                message=f"Currently busy with another job until {current_job_finish_time.isoformat()}",
                accepted=False,
            )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer_image: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/task_offer_image/",
        task_offer_image,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=str,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake)],
    )
    router.add_api_route(
        "/start_training/",
        tune_model_text,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_image/",
        tune_model_diffusion,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    return router