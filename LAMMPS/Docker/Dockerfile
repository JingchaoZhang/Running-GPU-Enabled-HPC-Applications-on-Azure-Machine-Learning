FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN apt update && apt-get install -y \
    build-essential \
    infiniband-diags \
    openssh-server \
    kmod \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Copy the NDv4 topology file
COPY ndv4-topo.xml .

# Compile LAMMPS
RUN wget https://github.com/lammps/lammps/archive/refs/tags/stable_2Aug2023_update2.tar.gz \
    && tar -xvf stable_2Aug2023_update2.tar.gz \
    && rm stable_2Aug2023_update2.tar.gz \
    && sed -i s/Volta70/AMPERE80/g /workspace/lammps-stable_2Aug2023_update2/src/MAKE/OPTIONS/Makefile.kokkos_cuda_mpi \
    && cd /workspace/lammps-stable_2Aug2023_update2/src/ \
    && make yes-kokkos \
    && make -j 12 kokkos_cuda_mpi

ENV PATH="/workspace/lammps-stable_2Aug2023_update2/src:${PATH}"