import os
import shutil
import uuid
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import docker
import toml
import yaml
from docker.errors import DockerException
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi

from core import constants as cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import save_config_toml
from core.config.config_handler import update_flash_attention
from core.config.config_handler import update_model_info
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.docker_utils import stream_logs
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import DiffusionJob
from core.models.utility_models import FileFormat
from core.models.utility_models import TextJob


logger = get_logger(__name__)

# Model-specific optimized configurations
MODEL_CONFIGS = {
    "llama": {
        "flash_attention": True,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate_scheduler": "cosine",
        "warmup_steps": 100,
        "eval_steps": 50,
        "save_steps": 100,
    },
    "mistral": {
        "flash_attention": True,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate_scheduler": "cosine",
        "warmup_steps": 100,
        "eval_steps": 50,
        "save_steps": 100,
    },
    "phi": {
        "flash_attention": False,  # Phi models may not support flash attention
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate_scheduler": "linear",
        "warmup_steps": 50,
        "eval_steps": 50,
        "save_steps": 100,
    },
    "qwen": {
        "flash_attention": True,  # Qwen2.5 supports flash attention
        "batch_size": 8,         # Smaller models can use larger batch sizes
        "gradient_accumulation_steps": 2,
        "learning_rate_scheduler": "cosine",
        "warmup_steps": 50,
        "eval_steps": 50,
        "save_steps": 100,
    },
    "samoline": {
        "flash_attention": True,  # Try flash attention for samoline models
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate_scheduler": "cosine",
        "warmup_steps": 100,
        "eval_steps": 50,
        "save_steps": 100,
        "load_in_8bit": False,   # Avoid quantization for unknown models
    },
    "sdxl": {
        "train_batch_size": 4,
        "max_train_steps": 1500,
        "learning_rate": 1e-4,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 100,
    },
    "stable-diffusion": {
        "train_batch_size": 8,
        "max_train_steps": 1000,
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 50,
    },
    "default": {
        "flash_attention": False,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate_scheduler": "cosine",
        "warmup_steps": 100,
        "eval_steps": 50,
        "save_steps": 100,
    }
}


@dataclass
class DockerEnvironmentDiffusion:
    huggingface_token: str
    wandb_token: str
    job_id: str

    def to_dict(self) -> dict[str, str]:
        return {"HUGGINGFACE_TOKEN": self.huggingface_token, "WANDB_TOKEN": self.wandb_token, "JOB_ID": self.job_id}


@dataclass
class DockerEnvironment:
    huggingface_token: str
    wandb_token: str
    job_id: str
    dataset_type: str
    dataset_filename: str

    def to_dict(self) -> dict[str, str]:
        return {
            "HUGGINGFACE_TOKEN": self.huggingface_token,
            "WANDB_TOKEN": self.wandb_token,
            "JOB_ID": self.job_id,
            "DATASET_TYPE": self.dataset_type,
            "DATASET_FILENAME": self.dataset_filename,
        }


def _get_model_family(model: str) -> str:
    """Get the model family for config lookup."""
    model_lower = model.lower()
    
    if "llama" in model_lower:
        return "llama"
    elif "mistral" in model_lower:
        return "mistral"
    elif "phi" in model_lower:
        return "phi"
    elif "qwen" in model_lower:
        return "qwen"
    elif "samoline" in model_lower:
        return "samoline"
    elif "stable-diffusion" in model_lower:
        return "stable-diffusion"
    elif "sdxl" in model_lower:
        return "sdxl"
    else:
        # Special handling for organization-based patterns
        splits = model_lower.split('/')
        if len(splits) > 1:
            org = splits[0]
            if org == "samoline":
                return "samoline"
    
        return "default"


def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
    task_id: str,
    expected_repo_name: str | None,
) -> dict:
    """
    Loads the config template and modifies it with optimized settings for the model.
    """
    logger.info("Loading config template with optimized settings")
    with open(cst.CONFIG_TEMPLATE_PATH, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    # Get the model family for optimization
    model_family = _get_model_family(model)
    model_config = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS["default"])
    
    # Apply model-specific optimizations
    config["flash_attention"] = model_config["flash_attention"]
    config["batch_size"] = model_config["batch_size"]
    config["gradient_accumulation_steps"] = model_config["gradient_accumulation_steps"]
    config["learning_rate_scheduler"] = model_config["learning_rate_scheduler"]
    config["train_on_inputs"] = True  # Better for finetuning
    config["warmup_steps"] = model_config["warmup_steps"]
    config["eval_steps"] = model_config["eval_steps"]
    config["save_steps"] = model_config["save_steps"]
    config["save_total_limit"] = 3  # Keep the 3 best checkpoints
    
    # Make sure we use proper settings based on model
    config = update_flash_attention(config, model)
    config = update_model_info(config, model, task_id, expected_repo_name)
    config["mlflow_experiment_name"] = dataset
    
    # Add model validation step to choose best checkpoint
    config["do_eval"] = True
    config["val_set_size"] = 0.05  # Use 5% of data for validation
    
    logger.info(f"Using optimized configuration for {model_family} model family")
    return config


def _load_and_modify_config_diffusion(model: str, task_id: str, expected_repo_name: str | None = None) -> dict:
    """
    Loads the config template and modifies it with optimized settings for the diffusion model.
    """
    logger.info("Loading diffusion config template with optimized settings")
    with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION, "r") as file:
        config = toml.load(file)
        
    # Get the model family for optimization
    model_family = _get_model_family(model)
    model_config = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS["default"])
    
    # Apply diffusion model-specific optimizations
    config["pretrained_model_name_or_path"] = model
    config["train_data_dir"] = f"/dataset/images/{task_id}/img/"
    config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
    config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{expected_repo_name or str(uuid.uuid4())}"
    
    # Apply optimizer settings based on model family
    if "train_batch_size" in model_config:
        config["train_batch_size"] = model_config["train_batch_size"]
    if "max_train_steps" in model_config:
        config["max_train_steps"] = model_config["max_train_steps"]
    if "learning_rate" in model_config:
        config["learning_rate"] = model_config["learning_rate"]
    if "lr_scheduler" in model_config:
        config["lr_scheduler"] = model_config["lr_scheduler"]
    if "lr_warmup_steps" in model_config:
        config["lr_warmup_steps"] = model_config["lr_warmup_steps"]
    
    logger.info(f"Using optimized diffusion configuration for {model_family} model family")
    return config


def create_job_diffusion(
    job_id: str,
    model: str,
    dataset_zip: str,
    expected_repo_name: str | None,
):
    return DiffusionJob(job_id=job_id, model=model, dataset_zip=dataset_zip, expected_repo_name=expected_repo_name)


def create_job_text(
    job_id: str,
    dataset: str,
    model: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
    expected_repo_name: str | None,
):
    return TextJob(
        job_id=job_id,
        dataset=dataset,
        model=model,
        dataset_type=dataset_type,
        file_format=file_format,
        expected_repo_name=expected_repo_name,
    )


def start_tuning_container_diffusion(job: DiffusionJob) -> Optional[Dict[str, Any]]:
    """Start a diffusion model tuning container with optimized settings.
    
    Returns evaluation metrics on success, None on failure.
    """
    logger.info("=" * 80)
    logger.info("STARTING THE DIFFUSION TUNING CONTAINER WITH OPTIMIZED SETTINGS")
    logger.info("=" * 80)
    
    start_time = time.time()
    config_path = os.path.join(cst.CONFIG_DIR, f"{job.job_id}.toml")

    config = _load_and_modify_config_diffusion(job.model, job.job_id, job.expected_repo_name)
    save_config_toml(config, config_path)

    logger.info(f"Using optimized config: {config}")

    # Prepare the dataset with improved settings
    prepare_dataset(
        training_images_zip_path=job.dataset_zip,
        training_images_repeat=cst.DIFFUSION_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=job.job_id,
    )

    docker_env = DockerEnvironmentDiffusion(
        huggingface_token=cst.HUGGINGFACE_TOKEN, wandb_token=cst.WANDB_TOKEN, job_id=job.job_id
    ).to_dict()
    logger.info(f"Docker environment: {docker_env}")

    metrics = None
    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/dataset/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/dataset/outputs",
                "mode": "rw",
            },
            os.path.abspath(cst.DIFFUSION_DATASET_DIR): {
                "bind": "/dataset/images",
                "mode": "rw",
            },
        }

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE_DIFFUSION,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
            detach=True,
            tty=True,
        )

        # Use the shared stream_logs function and capture metrics
        logs = stream_logs(container)
        
        # Try to extract metrics from logs
        metrics = _extract_diffusion_metrics(logs)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")
            
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully completed diffusion training in {elapsed_time:.2f} seconds")
        logger.info(f"Training metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        if "container" in locals():
            container.remove(force=True)

        train_data_path = f"{cst.DIFFUSION_DATASET_DIR}/{job.job_id}"

        if os.path.exists(train_data_path):
            shutil.rmtree(train_data_path)
            
    return metrics


def _extract_diffusion_metrics(logs: str) -> Dict[str, Any]:
    """Extract metrics from diffusion training logs."""
    metrics = {"loss": None}
    
    # Look for final loss in logs
    import re
    loss_matches = re.findall(r"loss: ([0-9.]+)", logs)
    if loss_matches:
        try:
            # Use the last reported loss
            metrics["loss"] = float(loss_matches[-1])
        except (ValueError, IndexError):
            pass
            
    return metrics


def start_tuning_container(job: TextJob) -> Optional[Dict[str, Any]]:
    """Start a text model tuning container with optimized settings.
    
    Returns evaluation metrics on success, None on failure.
    """
    logger.info("=" * 80)
    logger.info("STARTING THE TEXT TUNING CONTAINER WITH OPTIMIZED SETTINGS")
    logger.info("=" * 80)
    
    start_time = time.time()
    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)

    config = _load_and_modify_config(
        job.dataset,
        job.model,
        job.dataset_type,
        job.file_format,
        job.job_id,
        job.expected_repo_name,
    )
    save_config(config, config_path)

    logger.info(f"Using optimized config: {config}")

    logger.info(os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "")

    docker_env = DockerEnvironment(
        huggingface_token=cst.HUGGINGFACE_TOKEN,
        wandb_token=cst.WANDB_TOKEN,
        job_id=job.job_id,
        dataset_type=job.dataset_type.value if isinstance(job.dataset_type, DatasetType) else cst.CUSTOM_DATASET_TYPE,
        dataset_filename=os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "",
    ).to_dict()
    logger.info(f"Docker environment: {docker_env}")

    metrics = None
    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/workspace/axolotl/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/workspace/axolotl/outputs",
                "mode": "rw",
            },
        }

        if job.file_format != FileFormat.HF:
            dataset_dir = os.path.dirname(os.path.abspath(job.dataset))
            logger.info(dataset_dir)
            volume_bindings[dataset_dir] = {
                "bind": "/workspace/input_data",
                "mode": "ro",
            }

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
            detach=True,
            tty=True,
        )

        # Use the shared stream_logs function and capture logs
        logs = stream_logs(container)
        
        # Try to extract metrics from logs
        metrics = _extract_text_metrics(logs)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")
            
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully completed text model training in {elapsed_time:.2f} seconds")
        logger.info(f"Training metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        repo = config.get("hub_model_id", None)
        if repo:
            hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
            hf_api.update_repo_visibility(repo_id=repo, private=False, token=cst.HUGGINGFACE_TOKEN)
            logger.info(f"Successfully made repository {repo} public")

        if "container" in locals():
            try:
                container.remove(force=True)
                logger.info("Container removed")
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")
                
    return metrics


def _extract_text_metrics(logs: str) -> Dict[str, Any]:
    """Extract metrics from text model training logs."""
    metrics = {"loss": None, "perplexity": None}
    
    # Look for final validation loss in logs
    import re
    val_loss_matches = re.findall(r"val_loss: ([0-9.]+)", logs)
    if val_loss_matches:
        try:
            # Use the last reported validation loss
            metrics["loss"] = float(val_loss_matches[-1])
            # Calculate perplexity
            metrics["perplexity"] = 2 ** metrics["loss"]
        except (ValueError, IndexError):
            pass
            
    return metrics