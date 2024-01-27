#!/bin/bash

export UCX_TLS=tcp,cuda
export NCCL_TOPO_FILE=/workspace/ndv4-topo.xml
export UCX_NET_DEVICES=eth0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_SOCKET_IFNAME=eth0

lmp_kokkos_cuda_mpi \
    -k on g 8 -sf kk \
    -pk kokkos cuda/aware on neigh full comm device binsize 2.8 \
    -var x 16 -var y 4 -var z 8 -in in.lj