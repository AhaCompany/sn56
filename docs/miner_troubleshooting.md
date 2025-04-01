# G.O.D Miner Troubleshooting Guide

This guide provides detailed troubleshooting steps for common issues with the G.O.D (Gradients On Demand) miner. These improvements will help the miner complete training jobs successfully.

## Common Issues and Solutions

### 1. Environment Variables Not Set Properly

The miner requires several environment variables to be set correctly:

```bash
# Required variables
export HUGGINGFACE_TOKEN="your_hf_token_here"
export HUGGINGFACE_USERNAME="your_hf_username"

# Optional variables
export WANDB_TOKEN="your_wandb_token_here"  # For training visualization
export S3_BUCKET_NAME="your_s3_bucket"      # For S3 storage
```

**Solution**: 
- Create a `.env` file in the project root with these variables
- Run `source .env` before starting the miner
- Use the provided checker script: `python utils/check_miner_env.py`

### 2. Docker Issues

Common Docker-related problems:

- Docker service not running
- Missing Docker images
- Insufficient permissions
- NVIDIA Docker runtime not configured

**Solutions**:
1. Check Docker service: `systemctl status docker` or `docker info`
2. Ensure images are pulled:
   ```bash
   docker pull weightswandering/tuning_miner:latest
   docker pull diagonalge/diffusion_miner:latest
   ```
3. Add user to docker group: `sudo usermod -aG docker $USER`
4. For NVIDIA Docker: Install nvidia-docker2 and configure the runtime

### 3. GPU Access Issues

Problems with GPU access:

- Missing NVIDIA drivers
- GPU not detected by Docker
- Insufficient memory on GPU

**Solutions**:
1. Verify GPU is detected: `nvidia-smi`
2. Check Docker GPU access: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi`
3. Monitor GPU memory: `watch -n 1 nvidia-smi`
4. Adjust batch sizes in `job_handler.py` if memory is limited

### 4. Jobs Get Stuck in "Running" State

Jobs might get stuck due to:

- Docker container crashed
- Job timeout not triggered
- Watchdog thread issues
- Blocked stream_logs function

**Solutions**:
1. Check for zombie containers: `docker ps -a`
2. Clean up stale containers: `docker rm -f $(docker ps -aq)`
3. Configure timeout: Set `MAX_JOB_RUNTIME_SECONDS` env variable (default: 8 hours)
4. Restart the miner service to reset all threads

### 5. Dataset/Model Access Issues

Problems accessing datasets or models:

- Invalid dataset paths
- Hugging Face token issues
- Model family not supported
- Dataset formatting issues

**Solutions**:
1. Verify dataset exists and is correctly formatted
2. Check Hugging Face token has correct permissions
3. Make sure model family is in `ACCEPTED_MODEL_FAMILIES`
4. For local datasets, ensure paths are absolute and accessible

## Diagnostic Tools

### 1. Environment Checker

Run the environment checker to verify your setup:

```bash
python utils/check_miner_env.py
```

### 2. Container Logs

Check Docker container logs for specific error messages:

```bash
# List all containers
docker ps -a

# Get logs from a specific container
docker logs <container_id>
```

### 3. Miner Diagnostic Logs

The miner creates detailed diagnostic logs:

```bash
# View diagnostic logs
cat logs/miner_diagnostics.log

# Monitor logs in real-time
tail -f logs/miner_diagnostics.log
```

### 4. Container Inspection

Inspect Docker containers for configuration issues:

```bash
# Inspect container configuration
docker inspect <container_id>

# Check container resource usage
docker stats
```

## Performance Optimization

### 1. Model-Specific Configurations

The miner uses optimized configurations for different model families in `job_handler.py`. Adjust these settings if needed:

```python
MODEL_CONFIGS = {
    "llama": {
        "flash_attention": True,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        # ...
    },
    # ... other model families
}
```

### 2. Docker Resource Allocation

Adjust Docker resource limits if needed:

```bash
# Run with custom memory limits
docker run --gpus all --memory=16g --memory-swap=20g <image>
```

### 3. Priority Queue Settings

The miner uses a priority queue system to prioritize jobs for model families it handles well. Adjust the `PriorityJobQueue` settings in `training_worker.py` if needed.

## Complete Reset Procedure

If you encounter persistent issues, perform a complete reset:

1. Stop the miner service
2. Clean up Docker resources:
   ```bash
   docker stop $(docker ps -aq)
   docker rm $(docker ps -aq)
   ```
3. Clear temporary directories:
   ```bash
   rm -rf core/outputs/*
   rm -rf logs/*
   ```
4. Reset environment variables:
   ```bash
   source .env
   ```
5. Restart the miner service

## Enhanced Logging

The updated miner implementation includes improved logging:

1. Container logs are saved to `core/outputs/<job_id>_container.log`
2. Diagnostic logs are written to `logs/miner_diagnostics.log`
3. Sensitive information is masked in logs

## Running with Different Model Families

The miner now supports additional model families:

- llama
- mistral
- phi
- qwen (newly added)
- samoline (newly added)
- stable-diffusion
- sdxl

If adding support for new model families, update:
1. `_get_model_family` in `job_handler.py`
2. `MODEL_CONFIGS` in `job_handler.py`
3. `get_model_family` in `training_worker.py`
4. `ACCEPTED_MODEL_FAMILIES` in `endpoints/tuning.py`

## Contributing

When making changes to the miner codebase:

1. Add appropriate error handling
2. Include detailed logging
3. Validate inputs and environment variables
4. Test with different model families
5. Document any new configuration options