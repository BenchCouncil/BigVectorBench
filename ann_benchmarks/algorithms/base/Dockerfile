FROM ubuntu:22.04

# Change apt source to Tsinghua mirror
# RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu/g' /etc/apt/sources.list
# RUN apt-get clean

# Install Python3 and pip3
RUN apt-get update \
    && apt-get install -y python3 python3-pip wget \
    && pip3 install -U pip \
    && python3 --version
# Change pip source to Tsinghua mirror
# RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Set up the working directory
WORKDIR /home/app
# Install the required Python packages
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -r requirements.txt
# Copy the start-docker.sh script to the working directory
# This script is used to start the Docker service in the container
COPY ann_benchmarks/algorithms/base/start-docker.sh ./
RUN chmod +x start-docker.sh
ENTRYPOINT ["./start-docker.sh"]

# https://docs.docker.com/engine/install/ubuntu/
# Add Docker's official GPG key:
RUN apt-get update \
    && apt-get install -y ca-certificates curl \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null
# To install the latest version of Docker
RUN apt-get update \
    && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
