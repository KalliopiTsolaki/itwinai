# PoC for AI-centric digital twin workflows using Singularity containers

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)

See the latest version of our [docs](https://intertwin-eu.github.io/T6.5-AI-and-ML/)
for a quick overview of this platform for advanced AI/ML workflows in digital twin applications.

If you want to integrate a new use case, you can follow this
[step-by-step guide](https://intertwin-eu.github.io/T6.5-AI-and-ML/docs/How-to-use-this-software.html).

## CMCC Use-case:
To run do: 
```
micromamba run -p ./.venv python run-workflow.py -f ./use-cases/cyclones/workflows/workflow-train.yml
```

## Installation

The containers were build using Apptainer version 1.1.8-1.el8

### Building the containers

The container are built on top of the [NVIDIA PyTorch NGC Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). The NGC containers come with preinstalled libraries such as CUDA, cuDNN, NCCL, PyTorch, etc that are all harmouniously compatible with each other in order to reduce depenency issue and provide a maximum of portability. The current version used is ```nvcr.io/nvidia/pytorch:23.09-py3```, which is based on CUDA 12.2.1 and PyTorch 2.1.0a0+32f93b1.
If you need other specs you can consults the [Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) and find the right base container for you.
Once you found the right container you can alter the following line

```
apptainer pull torch.sif docker://nvcr.io/nvidia/pytorch:23.09-py3
```
inside ```containers/apptainer/apptainer_build.sh``` to change to the desired version.

As mentioned above additional libraries are installed on top of the NGC container which are listed inside ```env-files/torch/pytorch-env-gpu-container.txt```.

Once you are satisified with the libraries run:
```
./containers/apptainer/apptainer_build.sh
```

### Run the containers

Run the startscript with 
```
sbatch startscript.sh
```


### Future work
It is currently foreseen to build the container via GH actions.