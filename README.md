# Running GPU Enabled HPC Applications on Azure Machine Learning Platform
Azure Machine Learning (AML) is a cloud-based service primarily designed for machine learning applications. However, its versatile architecture also enables the deployment and execution of High-Performance Computing (HPC) applications. In this blog, we will explore AML's capabilities in running HPC applications, focusing on a setup involving two Standard_ND96asr_v4 SKU instances. This configuration harnesses the power of 16 NVIDIA A100 40G GPUs, providing a robust platform for demanding computational tasks. Our demonstration aims to showcase the effective utilization of AML not just for machine learning but also as a potent tool for running sophisticated HPC applications.

We will start with building an A100 GPU-based compute cluster within the AML environment. ollowing the cluster creation, we will proceed to configure the AML Environments, tailoring them specifically for running two key applications: the NCCL (NVIDIA Collective Communications Library) AllReduce Benchmark and the Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS). These applications will be executed across all 16 GPUs distributed over the two nodes. This exercise illustrats how to deploy and manage HPC applications using Azure Machine Learning. This approach is a departure from traditional methods typically reliant on SLURM for HPC resource management, highlighting AML's versatility and capability in handling complex HPC tasks.

# Create AML Cluster and Custom Environments
In this section, we will set up the AML infrastructure. The process involves creating an AML cluster and custom environments tailored for running HPC applications. This setup is crucial as it lays the foundational infrastructure and computational resources required for executing demanding tasks like the NCCL AllReduce Benchmark and LAMMPS simulations. We start by establishing a dedicated Resource Group and Workspace in Azure, followed by the construction of a Compute Cluster within AML. Finally, we create a specialized AML Environment, optimized for our specific HPC tasks. 

## Create Resource Group and AML WorkSpace
The Resource Group acts as a logical container for resources related to our project, ensuring organized management and easy tracking of Azure resources. Following this, we create an AML Workspace within this Resource Group. This Workspace serves as a central hub for all AML activities, including experiment management, data storage, and computational resource management. The code block demonstrates how to accomplish these tasks using Azure CLI commands, setting up the necessary infrastructure for our HPC applications.

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

# Define Docker image for the custom environment
env_name = "NCCL-Benchmark-Env"
custom_env = Environment(
    name=env_name,
    image='jzacr3.azurecr.io/pytorch_nccl_tests_2303:latest',
    version="1.0",
)

# Register the Environment
registered_env = ml_client.environments.create_or_update(custom_env)

# Print useful information about the environment
print(f"Environment '{env_name}' has been created/updated.")
print(f"Environment information: {registered_env}")
```

# NCCL AllReduce BW Test
The NCCL AllReduce test is a critical benchmark for assessing the communication performance of GPUs across nodes in a distributed computing environment. This test is particularly important for HPC applications as it provides insights into the efficiency and scalability of GPU interconnectivity, which is crucial for parallel processing tasks. We'll begin by submitting a job to run the NCCL AllReduce test on our previously configured AML environment and compute cluster, and then we will analyze the results to evaluate the performance.

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
