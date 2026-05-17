# Vehicles-Signal-Processing

Repository for Signal Processing in Automotive Engineering (FVE4012, Inha University) Final Project.

Our goal is to estimate time offset between sensors.

## Development Environment

This project is intended to run inside a Docker container while the source code remains in the local git repository.

The Docker environment is configured for:

- Ubuntu 22.04
- Python 3
- CPU-only PyTorch
- OpenCV
- Common scientific Python packages

CUDA is not used.

## Workspace Layout

Current local workspace layout:

```text
VSP_ws/
├── Dockerfile
├── docker.sh
├── requirements.txt
└── Vehicles-Signal-Processing/
    ├── README.md
    ├── LICENSE
    ├── test.py
    ├── datasets/
    ├── models/
    └── utils/
```

The local git repository is mounted into the Docker container:

```text
/home/jihun/Documents/VSP_ws/Vehicles-Signal-Processing
```

Container path:

```text
/workspace/Vehicles-Signal-Processing
```

## Build Docker Image

Build the Docker image from `VSP_ws/`:

```bash
cd /home/jihun/Documents/VSP_ws
sudo docker build -t vsp-image:latest .
```

## Run Docker Container

Run the container with:

```bash
./docker.sh
```

Inside the container, the working directory is:

```bash
/workspace/Vehicles-Signal-Processing
```

Example:

```bash
python test.py
```

## Current Dockerfile

```dockerfile
FROM ubuntu:22.04

#Set environment variables to prevent tzdata from prompting for geographic area
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

#필수 패키지 설치 및 Ninja 빌드 시스템 설치
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    git \
    build-essential \
    ninja-build \
    unzip \
    python3-pip \
    nano \
    python-is-python3

RUN apt-get install -y \
    cmake \
    libglib2.0-0 \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 설치
RUN pip3 install --no-cache-dir "networkx<3.0"
# 수정: CPU-only PyTorch
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    torchaudio==2.0.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# requirements.txt 복사
COPY requirements.txt /home/requirements.txt

# 필요한 패키지 설치
RUN /bin/bash -c "pip install --no-cache-dir -r /home/requirements.txt"

#OpenCV GUI 실행을 위한 X11/Qt dependency 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && \
    unzip opencv.zip

#작업 디렉토리 설정
WORKDIR /home
```

## Current docker.sh

```bash
#!/bin/bash

# sudo docker build -t vsp-image:latest .

xhost +local:docker
sudo docker run --name VSP-container -it \
  --privileged \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v /dev:/dev \
  -v /home/jihun/Documents/VSP_ws/Vehicles-Signal-Processing:/workspace/Vehicles-Signal-Processing \
  -w /workspace/Vehicles-Signal-Processing \
  -v "/media/jihun/Crucial X10:/ssd" \
  vsp-image:latest
```

## Current requirements.txt

```text
opencv-python
tqdm
tensorboard
addict
scikit-learn
pathspec
imagesize
ujson
cvxopt==1.2.7
matplotlib==3.5.0
numpy==1.22.0
pandas==1.3.5
Pyomo==6.4.2
PyYAML==6.0
scipy==1.7.3
Shapely==1.8.2
torchvision
Pillow
visdom
thop
```
