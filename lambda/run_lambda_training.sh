#!/bin/bash
# run_lambda_training.sh - Script to build and run LMFlow on Lambda GPU cluster
# Repository: git@github.com:lilyzhng/LMFlow.git
# Branch: data4elm-data-prep

# Configuration
DOCKER_IMAGE_NAME="lmflow-lambda"
CONTAINER_NAME="lmflow-training-$(date +%Y%m%d_%H%M%S)"
GITHUB_REPO="https://github.com/lilyzhng/LMFlow.git"
BRANCH="data4elm-data-prep"

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

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check for NVIDIA Docker support
if ! docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi &> /dev/null; then
    print_warning "NVIDIA Docker support may not be available. GPU training might not work."
fi

# Parse command line arguments
REBUILD=false
INTERACTIVE=true
WANDB_API_KEY=""
DATA_PATH=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --no-interactive)
            INTERACTIVE=false
            shift
            ;;
        --wandb-key)
            WANDB_API_KEY="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --rebuild              Force rebuild of Docker image"
            echo "  --no-interactive       Run in non-interactive mode"
            echo "  --wandb-key KEY        Set WandB API key"
            echo "  --data-path PATH       Mount custom data directory"
            echo "  --output-path PATH     Mount custom output directory"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default paths if not provided
if [[ -z "$OUTPUT_PATH" ]]; then
    OUTPUT_PATH="$(pwd)/output_models"
fi

if [[ -z "$DATA_PATH" ]]; then
    DATA_PATH="$(pwd)/data"
fi

# Create necessary directories
mkdir -p "$OUTPUT_PATH" "$DATA_PATH" "$(pwd)/log"

# Check if image exists and if rebuild is needed
if [[ "$REBUILD" == "true" ]] || ! docker image inspect ${DOCKER_IMAGE_NAME} &> /dev/null; then
    print_status "Building Docker image with latest LMFlow code from ${BRANCH} branch..."
    
    docker build \
        --build-arg GITHUB_REPO=${GITHUB_REPO} \
        --build-arg BRANCH=${BRANCH} \
        --build-arg USE_HTTPS=true \
        -t ${DOCKER_IMAGE_NAME} \
        -f Dockerfile.lambda \
        .

    if [ $? -ne 0 ]; then
        print_error "Docker build failed!"
        exit 1
    fi
    
    print_success "Docker image built successfully!"
else
    print_status "Using existing Docker image: ${DOCKER_IMAGE_NAME}"
fi

# Prepare Docker run command
DOCKER_CMD="docker run"
DOCKER_CMD+=" --name ${CONTAINER_NAME}"
DOCKER_CMD+=" --gpus all"
DOCKER_CMD+=" --shm-size=64g"
DOCKER_CMD+=" --ipc=host"
DOCKER_CMD+=" --ulimit memlock=-1"
DOCKER_CMD+=" --ulimit stack=67108864"

# Volume mounts
DOCKER_CMD+=" -v ${OUTPUT_PATH}:/workspace/LMFlow/output_models"
DOCKER_CMD+=" -v $(pwd)/log:/workspace/LMFlow/log"
DOCKER_CMD+=" -v ${DATA_PATH}:/workspace/LMFlow/data"

# Environment variables
DOCKER_CMD+=" -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
DOCKER_CMD+=" -e WANDB_PROJECT=lmflow-lambda-training"
DOCKER_CMD+=" -e WANDB_LOG_MODEL=checkpoint"

# Set WandB API key if provided
if [[ -n "$WANDB_API_KEY" ]]; then
    DOCKER_CMD+=" -e WANDB_API_KEY=${WANDB_API_KEY}"
    print_status "WandB API key configured"
else
    print_warning "WandB API key not provided. You'll need to login manually inside the container."
fi

# Interactive or detached mode
if [[ "$INTERACTIVE" == "true" ]]; then
    DOCKER_CMD+=" -it --rm"
    print_status "Starting interactive container..."
else
    DOCKER_CMD+=" -d"
    print_status "Starting container in detached mode..."
fi

DOCKER_CMD+=" ${DOCKER_IMAGE_NAME}"

# Add startup command for non-interactive mode
if [[ "$INTERACTIVE" == "false" ]]; then
    DOCKER_CMD+=" sleep infinity"
fi

# Run the container
print_status "Starting LMFlow container on Lambda GPU cluster..."
print_status "Container name: ${CONTAINER_NAME}"
print_status "Output directory: ${OUTPUT_PATH}"
print_status "Data directory: ${DATA_PATH}"

eval $DOCKER_CMD

if [ $? -eq 0 ]; then
    if [[ "$INTERACTIVE" == "true" ]]; then
        print_success "Container session completed!"
    else
        print_success "Container started successfully!"
        print_status "To attach to the container, run:"
        echo "  docker exec -it ${CONTAINER_NAME} bash"
        print_status "To view logs, run:"
        echo "  docker logs -f ${CONTAINER_NAME}"
        print_status "To stop the container, run:"
        echo "  docker stop ${CONTAINER_NAME}"
    fi
else
    print_error "Failed to start container!"
    exit 1
fi

# Show helpful commands
if [[ "$INTERACTIVE" == "true" ]]; then
    echo ""
    print_status "Inside the container, you can:"
    echo "  update-lmflow          # Update code from GitHub"
    echo "  setup-wandb            # Setup WandB (if API key not provided)"
    echo "  ./train_lambda.sh      # Start training with WandB logging"
    echo "  wandb login            # Login to WandB manually"
    echo ""
fi
