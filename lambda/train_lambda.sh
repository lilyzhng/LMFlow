#!/bin/bash
# train_lambda.sh - Training script optimized for Lambda GPU cluster with WandB integration
# Repository: git@github.com:lilyzhng/LMFlow.git
# Branch: data4elm-data-prep

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default parameters optimized for Lambda GPU cluster
model_name_or_path="data4elm/Llama-400M-12L"
dataset_path="nvidia/ClimbLab"
output_dir="/workspace/LMFlow/output_models/finetune"
num_gpus=$(nvidia-smi -L | wc -l)
deepspeed_args="--master_port=11000"
trust_remote_code=0

# WandB configuration
wandb_project="lmflow-lambda-training"
wandb_run_name=""
wandb_entity=""
wandb_tags=""
wandb_notes=""

# Training hyperparameters
num_train_epochs=1
learning_rate=1e-5
per_device_train_batch_size=8
gradient_accumulation_steps=4
block_size=1024
warmup_steps=100
logging_steps=10
save_steps=1000

# Parse arguments
while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_dir)
      output_dir="$2"
      shift
      ;;
    --num_gpus)
      num_gpus="$2"
      shift
      ;;
    --wandb_project)
      wandb_project="$2"
      shift
      ;;
    --wandb_run_name)
      wandb_run_name="$2"
      shift
      ;;
    --wandb_entity)
      wandb_entity="$2"
      shift
      ;;
    --wandb_tags)
      wandb_tags="$2"
      shift
      ;;
    --wandb_notes)
      wandb_notes="$2"
      shift
      ;;
    --num_train_epochs)
      num_train_epochs="$2"
      shift
      ;;
    --learning_rate)
      learning_rate="$2"
      shift
      ;;
    --per_device_train_batch_size)
      per_device_train_batch_size="$2"
      shift
      ;;
    --gradient_accumulation_steps)
      gradient_accumulation_steps="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    --disable_wandb)
      export WANDB_DISABLED=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Training Options:"
      echo "  -m, --model_name_or_path MODEL    Model name or path (default: data4elm/Llama-400M-12L)"
      echo "  -d, --dataset_path DATASET        Dataset path (default: nvidia/ClimbLab)"
      echo "  -o, --output_dir DIR              Output directory (default: /workspace/LMFlow/output_models/finetune)"
      echo "  --num_gpus NUM                    Number of GPUs to use (default: auto-detect)"
      echo "  --num_train_epochs NUM            Number of training epochs (default: 1)"
      echo "  --learning_rate RATE              Learning rate (default: 1e-5)"
      echo "  --per_device_train_batch_size NUM Batch size per device (default: 8)"
      echo "  --gradient_accumulation_steps NUM Gradient accumulation steps (default: 4)"
      echo "  --trust_remote_code 0|1           Trust remote code (default: 0)"
      echo ""
      echo "WandB Options:"
      echo "  --wandb_project PROJECT           WandB project name (default: lmflow-lambda-training)"
      echo "  --wandb_run_name NAME             WandB run name (default: auto-generated)"
      echo "  --wandb_entity ENTITY             WandB entity/team name"
      echo "  --wandb_tags TAGS                 WandB tags (comma-separated)"
      echo "  --wandb_notes NOTES               WandB run notes/description"
      echo "  --disable_wandb                   Disable WandB logging"
      echo ""
      echo "  -h, --help                        Show this help message"
      exit 0
      ;;
    *)
      print_error "Unknown option \"${key}\""
      echo "Use -h or --help for usage information"
      exit 1
  esac
  shift
done

# Validate GPU availability
if [ "$num_gpus" -eq 0 ]; then
    print_error "No GPUs detected! This script is designed for GPU training."
    exit 1
fi

# Setup directories
if [ -z "$wandb_run_name" ]; then
    exp_id="finetune_lambda_$(date +%Y%m%d_%H%M%S)"
    wandb_run_name="$exp_id"
else
    exp_id="$wandb_run_name"
fi

log_dir="/workspace/LMFlow/log/${exp_id}"
mkdir -p ${output_dir} ${log_dir}

# WandB setup and validation
print_status "Setting up WandB logging..."

if [ "$WANDB_DISABLED" != "true" ]; then
    # Check if WandB is logged in
    if ! wandb status &> /dev/null; then
        if [ -n "$WANDB_API_KEY" ]; then
            print_status "Logging into WandB with provided API key..."
            wandb login $WANDB_API_KEY
        else
            print_warning "WandB not logged in and no API key provided."
            print_warning "Run 'wandb login' or set WANDB_API_KEY environment variable."
            print_warning "Continuing with WandB disabled..."
            export WANDB_DISABLED=true
        fi
    else
        print_success "WandB is already logged in"
    fi
    
    # Set WandB environment variables
    export WANDB_PROJECT="$wandb_project"
    export WANDB_LOG_MODEL="checkpoint"
    export WANDB_WATCH="all"
    export WANDB_LOG_GRADIENTS=true
    export WANDB_LOG_PARAMETERS=true
    
    if [ -n "$wandb_entity" ]; then
        export WANDB_ENTITY="$wandb_entity"
    fi
    
    if [ -n "$wandb_tags" ]; then
        export WANDB_TAGS="$wandb_tags"
    fi
    
    if [ -n "$wandb_notes" ]; then
        export WANDB_NOTES="$wandb_notes"
    fi
    
    print_success "WandB configured for project: $wandb_project"
else
    print_warning "WandB logging is disabled"
fi

# Display training configuration
print_status "=== Training Configuration ==="
echo "Model: ${model_name_or_path}"
echo "Dataset: ${dataset_path}"
echo "Output Directory: ${output_dir}"
echo "Log Directory: ${log_dir}"
echo "Number of GPUs: ${num_gpus}"
echo "Epochs: ${num_train_epochs}"
echo "Learning Rate: ${learning_rate}"
echo "Batch Size per Device: ${per_device_train_batch_size}"
echo "Gradient Accumulation Steps: ${gradient_accumulation_steps}"
echo "Block Size: ${block_size}"
echo "WandB Project: ${wandb_project}"
echo "WandB Run Name: ${wandb_run_name}"
echo "WandB Disabled: ${WANDB_DISABLED:-false}"
print_status "================================"

# Determine DeepSpeed config based on number of GPUs
if [ "$num_gpus" -ge 4 ]; then
    deepspeed_config="configs/ds_config_zero2.json"
    print_status "Using ZeRO-2 configuration for ${num_gpus} GPUs"
elif [ "$num_gpus" -ge 2 ]; then
    deepspeed_config="configs/ds_config_zero2_no_offload.json"
    print_status "Using ZeRO-2 no offload configuration for ${num_gpus} GPUs"
else
    deepspeed_config="configs/ds_config_zero0_no_offload.json"
    print_status "Using ZeRO-0 configuration for ${num_gpus} GPU(s)"
fi

# Check if DeepSpeed config exists
if [ ! -f "$deepspeed_config" ]; then
    print_warning "DeepSpeed config $deepspeed_config not found, using default"
    deepspeed_config="configs/ds_config_zero2.json"
fi

print_status "Starting training on Lambda GPU cluster..."
print_status "Command will be logged to: ${log_dir}/train.log"

# Calculate effective batch size
effective_batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * num_gpus))
print_status "Effective batch size: ${effective_batch_size}"

# Run training with optimized settings for Lambda GPU cluster and WandB integration
deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${learning_rate} \
    --block_size ${block_size} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --use_dora 1 \
    --lora_r 16 \
    --lora_target_modules="embed_tokens,q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head" \
    --save_aggregated_lora 0 \
    --deepspeed ${deepspeed_config} \
    --bf16 \
    --run_name ${wandb_run_name} \
    --validation_split_percentage 0 \
    --logging_steps ${logging_steps} \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps ${save_steps} \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 32 \
    --gradient_checkpointing 1 \
    --warmup_steps ${warmup_steps} \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --report_to wandb \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err

# Check training completion status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Training completed successfully!"
    print_success "Model saved to: ${output_dir}"
    print_success "Logs saved to: ${log_dir}"
    
    if [ "$WANDB_DISABLED" != "true" ]; then
        print_success "Training metrics and plots are available in WandB:"
        print_success "Project: ${wandb_project}"
        print_success "Run: ${wandb_run_name}"
        print_success "Visit: https://wandb.ai/${WANDB_ENTITY:-your-entity}/${wandb_project}/runs/${wandb_run_name}"
    fi
else
    print_error "Training failed! Check logs at: ${log_dir}/train.err"
    exit 1
fi

# Display final summary
print_status "=== Training Summary ==="
echo "Start Time: $(head -1 ${log_dir}/train.log | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}' || echo 'N/A')"
echo "End Time: $(date)"
echo "Model: ${model_name_or_path}"
echo "Dataset: ${dataset_path}"
echo "Output: ${output_dir}"
echo "Logs: ${log_dir}"
echo "GPUs Used: ${num_gpus}"
echo "Effective Batch Size: ${effective_batch_size}"
if [ "$WANDB_DISABLED" != "true" ]; then
    echo "WandB Project: ${wandb_project}"
    echo "WandB Run: ${wandb_run_name}"
fi
print_status "========================"
