# Running GPU-Enabled HPC Applications on Azure Machine Learning Platform
Azure Machine Learning (AML) is a cloud-based service primarily designed for machine learning applications. However, its versatile architecture also enables the deployment and execution of High-Performance Computing (HPC) applications. In this blog, we explore AML's capabilities in running HPC applications, focusing on a setup involving two Standard_ND96asr_v4 SKU instances. This configuration harnesses the power of 16 NVIDIA A100 40G GPUs, providing a robust platform for demanding computational tasks. Our demonstration aims to showcase the effective utilization of AML not just for machine learning but also as a potent tool for running sophisticated HPC applications.

We will start with building an A100 GPU-based compute cluster within the AML environment. Following the cluster creation, we will proceed to configure the AML Environments, tailoring them specifically for running two key applications: the NCCL (NVIDIA Collective Communications Library) AllReduce Benchmark and the Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS). These applications will be executed across all 16 GPUs distributed over the two nodes. This exercise illustrats how to deploy and manage HPC applications using Azure Machine Learning. This approach is a departure from traditional methods typically reliant on SLURM for HPC resource management, highlighting AML's versatility and capability in handling complex HPC tasks. Scripts used in this blog can be found in this [GitHub repo](https://github.com/JingchaoZhang/Running-GPU-Enabled-HPC-Applications-on-Azure-Machine-Learning/tree/main).

# Parallel Backends in AML
Azure Machine Learning supports various parallel backends for distributed GPU training, each with its specific applications. The main frameworks are listed below:
- **Message Passing Interface**: The `MpiDistribution` method is used for distributed training with MPI. It requires a base Docker image with an MPI library, with OpenMPI included in all AML GPU base images.
- **PyTorch**: AML supports PyTorch's native distributed training capabilities (`torch.distributed`) using the nccl backend for GPU-based training. AML sets the necessary environment variables for process group initialization and does not require a separate launcher utility like torch.distributed.launch.
- **TensorFlow**: For native distributed TensorFlow, such as TensorFlow 2.x's tf.distribute.Strategy API, AML supports launching distributed jobs using the distribution parameters or `TensorFlowDistribution` object. AML automatically configures the `TF_CONFIG` environment variable for distributed TensorFlow jobs.

For High-Performance Computing (HPC) applications, we employed the `MpiDistribution` method, which is particularly effective for tasks requiring high inter-node communication efficiency. Note that at the moment of this blog, AML constructs the full MPI launch command behind the scenes. You can't provide your own full head-node-launcher commands like `mpirun`.

# Create AML Cluster and Custom Environments
We start by establishing a dedicated Resource Group and Workspace in Azure, followed by the construction of a Compute Cluster within AML. Then, we create a specialized AML Environment, optimized for our specific HPC tasks. 

## Create Resource Group and AML WorkSpace
The Resource Group acts as a logical container for resources related to our project, ensuring organized management and easy tracking of Azure resources. The Workspace serves as a central hub for all AML activities, including experiment management, data storage, and computational resource management. The code block demonstrates how to accomplish these tasks using Azure CLI commands, setting up the necessary infrastructure for our HPC applications.

```bash
export RG=${ResourceGroup}
export location=southcentralus
export ws_name=${WorkSpace}
export SubID=$(az account show --query id -o tsv)
export CLUSTER_NAME="NDv4"
export acr_id="/subscriptions/${SubID}/resourceGroups/${ResourceGroup}/providers/Microsoft.ContainerRegistry/registries/${ACR}"

az group create --name $RG --location $location
az ml workspace create --name $ws_name --resource-group $RG --location $location --container-registry $acr_id
```

## Create Compute Cluster in AML
The first step in our setup involves creating a Resource Group and an AML Workspace. The Resource Group acts as a logical container for resources related to our project, ensuring organized management and easy tracking of Azure resources. Following this, we create an AML Workspace within this Resource Group. This Workspace serves as a central hub for all AML activities, including experiment management, data storage, and computational resource management. The code block demonstrates how to accomplish these tasks using Azure CLI commands, setting up the necessary infrastructure for our HPC applications.

```python
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import IdentityConfiguration,AmlCompute,AmlComputeSshSettings
from azure.ai.ml.constants import ManagedServiceIdentityType

# Retrieve details from environment variables
subscription_id = os.getenv('SubID')
resource_group = os.getenv('RG')
work_space = os.getenv('ws_name')
cluster_name = os.getenv('CLUSTER_NAME')

# get a handle to the workspace
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, work_space)

# Create an identity configuration for a system-assigned managed identity
identity_config = IdentityConfiguration(type = ManagedServiceIdentityType.SYSTEM_ASSIGNED)

cluster_ssh = AmlComputeSshSettings(
    admin_username="azureuser",
    ssh_key_value="${PublicKey}",
    admin_password="${PassWord}",)

cluster_basic = AmlCompute(
    name=cluster_name,
    type="amlcompute",
    size="Standard_ND96asr_v4",
    ssh_public_access_enabled=True,
    ssh_settings=cluster_ssh,
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120,
    identity=identity_config)

operation = ml_client.begin_create_or_update(cluster_basic)
result = operation.result()

# Print useful information about the operation
print(f"Cluster '{cluster_name}' has been created/updated.")
print(f"Cluster information: {result}")
```

## Create Custom AML Environment
This environment is designed to provide the necessary runtime context for our HPC applications, including the NCCL AllReduce Benchmark. The Python code in this section uses the Azure AI ML SDK to define and register a custom environment using a specific Docker image. This image is pre-configured with the required dependencies and frameworks, ensuring a consistent and optimized execution environment across all computations within the AML framework.

```python
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

# Retrieve details from environment variables
subscription_id = os.getenv('SubID')
resource_group = os.getenv('RG')
work_space = os.getenv('ws_name')
cluster_name = os.getenv('CLUSTER_NAME')

# get a handle to the workspace
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, work_space)

# Define Docker image for the NCCL custom environment
env_name = "NCCL-Benchmark-Env"
NCCL_env = Environment(
    name=env_name,
    image='jzacr3.azurecr.io/pytorch_nccl_tests_2303:latest',
    version="1.0",
)

# Register the Environment
registered_env = ml_client.environments.create_or_update(NCCL_env)

# Print useful information about the environment
print(f"Environment '{env_name}' has been created/updated.")
print(f"Environment information: {registered_env}")

# Define Docker image for LAMMPS custom environment
env_name = "LAMMPS-Benchmark-Env"
LAMMPS_env = Environment(
    name=env_name,
    #image='nvcr.io/hpc/lammps:patch_15Jun2023',
    image='jzacr3.azurecr.io/nvhpc_lammps:latest',
    version="2.0",
)

# Register the Environment
registered_env = ml_client.environments.create_or_update(LAMMPS_env)

# Print useful information about the environment
print(f"Environment '{env_name}' has been created/updated.")
print(f"Environment information: {registered_env}")
```

# NCCL AllReduce BW Test
The NCCL AllReduce test is a critical benchmark for assessing the communication performance of GPUs across nodes in a distributed computing environment. This test is particularly important for HPC applications as it provides insights into the efficiency and scalability of GPU interconnectivity, which is crucial for parallel processing tasks. We begin by submitting a job to run the NCCL AllReduce test on our previously configured AML environment and compute cluster, and then we will analyze the results to evaluate the performance.

## Submit NCCL AllReduce Job
To execute the NCCL AllReduce Bandwidth Test, we first need to submit a job to our Azure ML compute cluster. The Python code shown here is designed to accomplish this task. It uses the Azure AI ML SDK to define and submit a command job. This job runs a bash script (NCCL.sh) located in the specified source directory. The script is responsible for initiating the NCCL AllReduce test. The job configuration includes details such as the compute target (the name of our AML compute cluster), the environment (our custom NCCL benchmark environment), and the distribution setting, which is set to use MPI for parallel processing across multiple instances. Additionally, services like JupyterLab are configured for interactive job monitoring and analysis.

```python
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient,command,MpiDistribution

from azure.ai.ml.entities import (
    SshJobService,
    VsCodeJobService,
    JupyterLabJobService,
)

# Retrieve details from environment variables
subscription_id = os.getenv('SubID')
resource_group = os.getenv('RG')
work_space = os.getenv('ws_name')
cluster_name = os.getenv('CLUSTER_NAME')

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, work_space)

job = command(
    code="./src",  # local path where the code is stored
    command="bash NCCL.sh",
    compute=cluster_name,
    environment="NCCL-Benchmark-Env:1.0",
    instance_count=2,
    distribution=MpiDistribution(
        process_count_per_instance=8,
    ),
    services={
        "My_jupyterlab": JupyterLabJobService(),
    }
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
```

## NCCL AllReduce Result
After submitting the NCCL AllReduce test job, we obtain a series of results that provide insights into the bandwidth and communication performance of the GPUs. The output below shows the optimal performance of around 188 GB/s between two `Standard_ND96asr_v4` nodes.
```bash
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    37.76    0.00    0.00      0    36.09    0.00    0.00      0
          16             4     float     sum      -1    37.45    0.00    0.00      0    37.07    0.00    0.00      0
          32             8     float     sum      -1    36.38    0.00    0.00      0    35.53    0.00    0.00      0
          64            16     float     sum      -1    37.00    0.00    0.00      0    36.36    0.00    0.00      0
         128            32     float     sum      -1    36.50    0.00    0.01      0    35.26    0.00    0.01      0
         256            64     float     sum      -1    37.62    0.01    0.01      0    36.90    0.01    0.01      0
         512           128     float     sum      -1    38.44    0.01    0.02      0    37.76    0.01    0.03      0
        1024           256     float     sum      -1    40.18    0.03    0.05      0    39.85    0.03    0.05      0
        2048           512     float     sum      -1    44.38    0.05    0.09      0    42.45    0.05    0.09      0
        4096          1024     float     sum      -1    47.48    0.09    0.16      0    46.14    0.09    0.17      0
        8192          2048     float     sum      -1    64.17    0.13    0.24      0    53.94    0.15    0.28      0
       16384          4096     float     sum      -1    105.5    0.16    0.29      0    57.07    0.29    0.54      0
       32768          8192     float     sum      -1    65.22    0.50    0.94      0    56.87    0.58    1.08      0
       65536         16384     float     sum      -1    73.32    0.89    1.68      0    56.58    1.16    2.17      0
      131072         32768     float     sum      -1    66.52    1.97    3.69      0    63.45    2.07    3.87      0
      262144         65536     float     sum      -1    68.98    3.80    7.13      0    67.47    3.89    7.28      0
      524288        131072     float     sum      -1    79.28    6.61   12.40      0    79.27    6.61   12.40      0
     1048576        262144     float     sum      -1    96.57   10.86   20.36      0    96.62   10.85   20.35      0
     2097152        524288     float     sum      -1    127.5   16.45   30.85      0    127.9   16.39   30.74      0
     4194304       1048576     float     sum      -1    147.0   28.53   53.49      0    147.4   28.46   53.37      0
     8388608       2097152     float     sum      -1    210.7   39.82   74.65      0    209.2   40.10   75.18      0
    16777216       4194304     float     sum      -1    332.4   50.48   94.65      0    331.1   50.67   95.00      0
    33554432       8388608     float     sum      -1    614.9   54.57  102.32      0    621.0   54.03  101.30      0
    67108864      16777216     float     sum      -1    947.8   70.81  132.76      0    938.9   71.48  134.02      0
   134217728      33554432     float     sum      -1   1682.8   79.76  149.55      0   1689.0   79.47  149.00      0
   268435456      67108864     float     sum      -1   2975.4   90.22  169.16      0   3000.5   89.46  167.74      0
   536870912     134217728     float     sum      -1   5744.6   93.46  175.23      0   5693.8   94.29  176.80      0
  1073741824     268435456     float     sum      -1    11080   96.91  181.70      0    11095   96.78  181.46      0
  2147483648     536870912     float     sum      -1    21872   98.19  184.10      0    21765   98.66  185.00      0
  4294967296    1073741824     float     sum      -1    43176   99.48  186.52      0    43174   99.48  186.52      0
  8589934592    2147483648     float     sum      -1    85820  100.09  187.67      0    85895  100.01  187.51      0
f1d5e1df5ed649d0b9e618783ea5fdbc000000:1211:1211 [0] NCCL INFO comm 0x5634cae893b0 rank 0 nranks 16 cudaDev 0 busId 100000 - Destroy COMPLETE
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 57.1241 
```

# Distributed LAMMPS Job
LAMMPS is a widely used molecular dynamics simulation software. This distributed approach is particularly beneficial for simulations requiring extensive computational resources, as it allows for parallel processing across multiple GPUs, thereby significantly enhancing performance and reducing runtime.

## Submit GPU Enabled LAMMPS on Two `Standard_ND96asr_v4` Nodes
The job configuration includes the path to the LAMMPS executable script (run_lammps.sh), the compute target, and the custom environment specifically set up for LAMMPS. Additionally, the job is configured to run on two nodes, each utilizing 8 MPI processes, to leverage the distributed computing capabilities. The script also specifies shared memory size and includes services for JupyterLab and SSH, enabling interactive analysis and remote access to the running job for monitoring and debugging purposes.

```bash
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient,command,MpiDistribution
from azure.ai.ml.entities import JupyterLabJobService, SshJobService

# Retrieve details from environment variables
subscription_id = os.getenv('SUBSCRIPTION_ID')
resource_group = os.getenv('RESOURCE_GROUP')
work_space = os.getenv('WORKSPACE_NAME')
cluster_name = os.getenv('CLUSTER_NAME')

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, work_space)

job = command(
    code="./src",  # local path where the code is stored
    command="bash run_lammps.sh",
    compute=cluster_name,
    environment="LAMMPS-Benchmark-Env:2.0",
    instance_count=2,
    shm_size="2g",
    distribution=MpiDistribution(
        process_count_per_instance=8,
    ),
    services={
        "My_jupyterlab": JupyterLabJobService(),
        "My_ssh": SshJobService(
                ssh_public_keys=${PublicKey},
                nodes="all"),
    }
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
```

The output concludes with a summary of the computational performance, including the total wall time and a breakdown of the time spent in various sections of the code.
## LAMMPS Output
```bash
LAMMPS (2 Aug 2023 - Update 2)
KOKKOS mode with Kokkos version 3.7.2 is enabled (../kokkos.cpp:108)
  will use up to 8 GPU(s) per node
Lattice spacing in x,y,z = 1.6795962 1.6795962 1.6795962
Created orthogonal box = (0 0 0) to (537.47078 134.3677 268.73539)
  4 by 1 by 4 MPI processor grid
Created 16384000 atoms
  using lattice units in orthogonal box = (0 0 0) to (537.47078 134.3677 268.73539)
  create_atoms CPU = 0.140 seconds
Generated 0 of 0 mixed pair_coeff terms from geometric mixing rule
Neighbor list info ...
  update: every = 20 steps, delay = 0 steps, check = no
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 2.8
  ghost atom cutoff = 2.8
  binsize = 2.8, bins = 192 48 96
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair lj/cut/kk, perpetual
      attributes: full, newton off, kokkos_device
      pair build: full/bin/kk/device
      stencil: full/bin/3d
      bin: kk/device
Setting up Verlet run ...
  Unit style    : lj
  Current step  : 0
  Time step     : 0.005
Per MPI rank memory allocation (min/avg/max) = 155.7 | 157.1 | 158 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press     
         0   1.44          -6.7733681      0             -4.6133682     -5.0196693    
       100   0.75944938    -5.7614943      0             -4.6223203      0.18979435   
       200   0.75691889    -5.7579983      0             -4.6226201      0.22453779   
       300   0.74862879    -5.7454254      0             -4.6224823      0.30320043   
       400   0.73952934    -5.7315134      0             -4.6222195      0.38830386   
       500   0.73109398    -5.7185411      0             -4.6219002      0.46601068   
       600   0.72319764    -5.7063472      0             -4.6215508      0.53734812   
       700   0.71627778    -5.6956228      0             -4.6212062      0.59840289   
       800   0.71088121    -5.6872216      0             -4.6208999      0.64548402   
       900   0.70681548    -5.680866       0             -4.6206428      0.67980724   
      1000   0.7037414     -5.6760744      0             -4.6204623      0.70559832   
Loop time of 11.1074 on 16 procs for 1000 steps with 16384000 atoms

Performance: 38892.844 tau/day, 90.030 timesteps/s, 1.475 Gatom-step/s
47.9% CPU use with 16 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.046821   | 0.049583   | 0.051841   |   0.5 |  0.45
Neigh   | 0.24714    | 0.2909     | 0.32389    |   3.8 |  2.62
Comm    | 8.6846     | 8.7882     | 9.0226     |   2.6 | 79.12
Output  | 0.02551    | 0.236      | 0.45515    |  29.1 |  2.12
Modify  | 1.1376     | 1.3388     | 1.4242     |   5.6 | 12.05
Other   |            | 0.4039     |            |       |  3.64

Nlocal:      1.024e+06 ave 1.02447e+06 max 1.02358e+06 min
Histogram: 1 1 1 3 2 4 2 1 0 1
Nghost:         180593 ave      180826 max      180297 min
Histogram: 1 0 0 1 4 3 4 2 0 1
Neighs:              0 ave           0 max           0 min
Histogram: 16 0 0 0 0 0 0 0 0 0
FullNghs:  7.67727e+07 ave 7.68284e+07 max 7.67214e+07 min
Histogram: 2 0 1 1 4 3 3 1 0 1

Total # of neighbors = 1.2283633e+09
Ave neighs/atom = 74.973344
Neighbor list builds = 50
Dangerous builds not checked
Total wall time: 0:00:14
```

# Conclusion
- On an AML compute cluster with two `Standard_ND96asr_v4` nodes, we achieved optimal NCCL AllReduce IB performance reaching up to 188 GB/s, which is pivotal for running tightly-coupled cross-node GPU applications on AML.
- AML's capability extends beyond machine learning applications. It proves to be a powerful platform for GPU enabled HPC tasks, as evidenced by NCCL AllReduce Benchmark and LAMMPS.
- The creation of custom AML environments, tailored to specific HPC applications, showcases the flexibility of Azure ML in providing optimized runtime contexts.
- The NCCL AllReduce Bandwidth Test and the distributed LAMMPS simulation illustrate AML's scalability and robust performance in distributed computing environments. The ability to efficiently parallel-process across multiple GPU nodes is a testament to AML's suitability for large-scale HPC applications.

# References
- [Running AutoDock HPC application on Azure Machine Learning platform](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/running-autodock-hpc-application-on-azure-machine-learning/ba-p/4020180)
- [AML Distributed GPU Training](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-distributed-gpu?view=azureml-api-2)
- [AML Manage environments with SDK and CLI (v2)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?view=azureml-api-2&tabs=cli)
- [AML debug-and-monitor](https://github.com/Azure/azureml-examples/blob/dd15e3f7d6a512fedfdfbdb4be19e065e8c1d224/sdk/python/jobs/single-step/debug-and-monitor/debug-and-monitor.ipynb)
- [AML Compute Class](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.amlcompute?view=azure-python)
