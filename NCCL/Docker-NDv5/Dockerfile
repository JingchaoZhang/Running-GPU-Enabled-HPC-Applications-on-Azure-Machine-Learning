FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN apt-get update && apt-get install -y \
    build-essential \
    infiniband-diags \
    openssh-server \
    kmod \
    && rm -rf /var/lib/apt/lists/*

COPY nccl-tests.sh .
COPY ndv5-topo.xml .

RUN chmod +x nccl-tests.sh && ./nccl-tests.sh
