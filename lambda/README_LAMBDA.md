# LMFlow Lambda GPU Cluster Setup

This guide explains how to run LMFlow training on Lambda GPU cluster with automatic GitHub code pulling and WandB visualization.

## Files Overview

- `Dockerfile.lambda` - Docker configuration for Lambda GPU cluster
- `run_lambda_training.sh` - Script to build and run Docker container
- `train_lambda.sh` - Training script with WandB integration

## Prerequisites

1. **Lambda GPU Cluster Access**: Ensure you have access to Lambda GPU cluster
2. **Docker with GPU Support**: Installed on the Lambda cluster
3. **WandB Account**: For training visualization (optional but recommended)

## Quick Start

### 1. Setup WandB (Recommended)

Get your WandB API key from [https://wandb.ai/settings](https://wandb.ai/settings)

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

### 2. Run Training

```bash
# Build and run with WandB integration
./run_lambda_training.sh --wandb-key $WANDB_API_KEY

# Inside the container, start training
./train_lambda.sh
```

## Detailed Usage

### Building and Running Container

```bash
# Basic usage
./run_lambda_training.sh

# With custom options
./run_lambda_training.sh \
    --rebuild \
    --wandb-key "your_api_key" \
    --data-path "/path/to/your/data" \
    --output-path "/path/to/outputs"

# Non-interactive mode (for automated runs)
./run_lambda_training.sh --no-interactive
```

### Training Options

```bash
# Basic training with default settings
./train_lambda.sh

# Custom training configuration
./train_lambda.sh \
    --model_name_or_path "microsoft/DialoGPT-medium" \
    --dataset_path "your/dataset" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --wandb_project "my-lmflow-experiment" \
    --wandb_run_name "experiment-v1"

# Training without WandB
./train_lambda.sh --disable_wandb
```

### Available Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name_or_path` | Model name or path | `data4elm/Llama-400M-12L` |
| `--dataset_path` | Dataset path | `nvidia/ClimbLab` |
| `--output_dir` | Output directory | `/workspace/LMFlow/output_models/finetune` |
| `--num_train_epochs` | Number of epochs | `1` |
| `--learning_rate` | Learning rate | `1e-5` |
| `--per_device_train_batch_size` | Batch size per GPU | `8` |
| `--gradient_accumulation_steps` | Gradient accumulation | `4` |

### WandB Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--wandb_project` | WandB project name | `lmflow-lambda-training` |
| `--wandb_run_name` | Run name | Auto-generated |
| `--wandb_entity` | WandB team/entity | - |
| `--wandb_tags` | Tags (comma-separated) | - |
| `--wandb_notes` | Run description | - |
| `--disable_wandb` | Disable WandB logging | - |

## WandB Visualization

Once training starts, you can monitor:

1. **Training Loss**: Real-time loss curves
2. **Learning Rate**: Learning rate schedule
3. **GPU Utilization**: Resource usage metrics
4. **Model Checkpoints**: Saved model artifacts
5. **System Metrics**: Memory, GPU temperature, etc.

Visit your WandB dashboard at: `https://wandb.ai/your-entity/lmflow-lambda-training`

## Container Management

### Useful Commands Inside Container

```bash
# Update code from GitHub
update-lmflow

# Setup WandB (if not done during container start)
setup-wandb

# Check GPU availability
nvidia-smi

# Monitor training logs
tail -f log/your_experiment/train.log
```

### External Container Management

```bash
# List running containers
docker ps

# Attach to running container
docker exec -it lmflow-training-YYYYMMDD_HHMMSS bash

# View container logs
docker logs -f lmflow-training-YYYYMMDD_HHMMSS

# Stop container
docker stop lmflow-training-YYYYMMDD_HHMMSS
```

## File Structure

```
/workspace/LMFlow/
├── output_models/          # Model outputs (mounted)
├── log/                   # Training logs (mounted)
├── data/                  # Training data (mounted)
├── examples/              # Training scripts
├── configs/               # DeepSpeed configurations
└── src/lmflow/           # LMFlow source code
```

## Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   nvidia-smi  # Check GPU status
   docker run --gpus all nvidia/cuda:11.8-base nvidia-smi  # Test Docker GPU
   ```

2. **WandB Login Issues**
   ```bash
   wandb login  # Manual login
   export WANDB_API_KEY="your_key"  # Set API key
   ```

3. **Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable gradient checkpointing (already enabled)

4. **Container Build Issues**
   ```bash
   ./run_lambda_training.sh --rebuild  # Force rebuild
   docker system prune  # Clean up Docker cache
   ```

### Performance Optimization

1. **Multi-GPU Training**: Automatically detected and configured
2. **Memory Optimization**: DeepSpeed ZeRO configurations based on GPU count
3. **Data Loading**: Optimized number of workers
4. **Mixed Precision**: BF16 enabled for faster training

## Repository Information

- **Repository**: `git@github.com:lilyzhng/LMFlow.git`
- **Branch**: `data4elm-data-prep`
- **Auto-Update**: Code is automatically pulled during container build

## Support

For issues specific to:
- **LMFlow**: Check the main repository issues
- **Lambda GPU Cluster**: Contact Lambda support
- **WandB**: Check WandB documentation

## Advanced Usage

### Custom DeepSpeed Configuration

The training script automatically selects DeepSpeed configuration based on GPU count:
- 1 GPU: `ds_config_zero0_no_offload.json`
- 2-3 GPUs: `ds_config_zero2_no_offload.json`
- 4+ GPUs: `ds_config_zero2.json`

### Environment Variables

Set these in your shell or container:

```bash
export WANDB_PROJECT="my-project"
export WANDB_ENTITY="my-team"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM=false
```

### Automated Training Pipeline

For automated runs:

```bash
# Start container in detached mode
./run_lambda_training.sh --no-interactive --wandb-key $WANDB_API_KEY

# Run training in background
docker exec -d lmflow-training-YYYYMMDD_HHMMSS ./train_lambda.sh

# Monitor progress
docker logs -f lmflow-training-YYYYMMDD_HHMMSS
```
