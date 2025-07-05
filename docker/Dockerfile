FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04

ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y \
    fontconfig \
    libopenmpi-dev \
    git \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    build-essential \
    cmake \
    curl \
    wget \
    --fix-missing && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.9 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 11.8 support FIRST
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install MPI4PY
RUN pip install mpi4py

# Set working directory
WORKDIR /LMFlow/

# Copy requirements and modify them to exclude torch (since we installed it with specific CUDA version)
COPY requirements.txt /LMFlow/requirements_temp.txt
RUN grep -v "^torch" /LMFlow/requirements_temp.txt > /LMFlow/requirements_filtered.txt

# Install filtered requirements
RUN pip install -r requirements_filtered.txt

# Install Jupyter and related packages
RUN pip install \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    tqdm \
    ipykernel

# Copy source code
ARG SRCDIR=.
COPY $SRCDIR/ /LMFlow/

# Install the package in development mode
RUN pip install -e .

# Set environment variables for better performance
ENV TOKENIZERS_PARALLELISM=false
ENV WANDB_CACHE_DIR=/tmp/wandb
ENV CUDA_VISIBLE_DEVICES=0

# Create necessary directories
RUN mkdir -p /tmp/wandb /tmp/transformers_cache /LMFlow/notebooks

# Set proper permissions
RUN chmod -R 755 /LMFlow/

# Configure Jupyter
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose Jupyter port
EXPOSE 8888

# Set default command
CMD ["bash"]