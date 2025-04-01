# G.O.D Custom Miner for Maximizing Validator Scores

This is a custom implementation of the G.O.D subnet miner, optimized to achieve the highest possible scores from validators.

## Key Optimizations

### 1. Intelligent Job Queue Management
- **Priority-based Queue**: Jobs are prioritized based on model family expertise
- **Historical Performance Tracking**: The system learns which model families it performs well with
- **Failure Avoidance**: Jobs for model families that have failed in the past are rejected
- **Parallel Processing**: Can handle multiple jobs concurrently (configurable)

### 2. Model-Specific Training Optimization
- **Per-model Family Configurations**: Customized training parameters for different model types:
  - Llama models use flash attention and cosine learning rate scheduler
  - Mistral models use similar optimizations tailored to their architecture
  - Phi models use different batch sizes and linear learning rate scheduler
  - Diffusion models have type-specific optimizations (SDXL vs Stable Diffusion)

### 3. Smart Job Acceptance Criteria
- **Model Family Filtering**: Only accepts jobs for model families it can handle well
- **Queue Management**: Controls queue size to avoid overwhelming the system
- **Time Constraints**: Sets reasonable limits on job completion time
- **Environment Variable Configuration**: Easily configurable via environment variables:
  ```
  # Job limits
  MAX_TEXT_TASK_HOURS=12
  MAX_IMAGE_TASK_HOURS=3
  
  # Supported model families (comma-separated)
  ACCEPTED_MODEL_FAMILIES=llama,mistral
  
  # Multi-processing
  MINER_MAX_CONCURRENT_JOBS=1
  
  # Job queue size
  MAX_QUEUE_SIZE=5
  ```

### 4. Enhanced Training Process
- **Validation Set Integration**: Automatically reserves 5% of data for validation
- **Best Checkpoint Selection**: Saves multiple checkpoints and selects the best one
- **Metrics Extraction**: Captures and logs performance metrics from training
- **Optimized Hyperparameters**: Pre-configured settings for different model types

## Usage

1. Set your environment variables in `.env` file
2. Start the miner with `task miner`
3. Monitor logs to see job acceptance and processing

## Expected Results

This custom miner implementation should significantly improve your scores by:
1. Only accepting jobs it can reliably complete
2. Optimizing training for different model architectures
3. Using best practices for fine-tuning each model type
4. Prioritizing jobs where it has proven expertise
5. Learning from past performance to continuously improve

The system will automatically adapt to your hardware capabilities and focus on the model families where it achieves the best results, maximizing your validator scores over time.