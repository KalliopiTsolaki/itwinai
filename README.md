# PoC for AI-centric digital twin workflows using Singularity containers

[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Super-Linter](https://github.com/interTwin-eu/T6.5-AI-and-ML/actions/workflows/check-links.yml/badge.svg)](https://github.com/marketplace/actions/markdown-link-check)

See the latest version of our [docs](https://intertwin-eu.github.io/T6.5-AI-and-ML/)
for a quick overview of this platform for advanced AI/ML workflows in digital twin applications.

If you want to integrate a new use case, you can follow this
[step-by-step guide](https://intertwin-eu.github.io/T6.5-AI-and-ML/docs/How-to-use-this-software.html).

<<<<<<< HEAD
### Requirements
=======
## CMCC Use-case:
To run do: 
```
micromamba run -p ./.venv python run-workflow.py -f ./use-cases/cyclones/workflows/workflow-train.yml
```

## Installation
>>>>>>> 19cf110 (Update README.md)

The containers were build using Singularity 3.11.4. 

### Building the containers

Currently, there are two containers, which are created manually. 
- preprocessing
- ai

They can be built via executing bash scripts from the root directory of this repo:

```
sudo ./ai/containers/build-base-container.sh
```

and

```
sudo ./use-cases/mnist/containers/build-preproc-container.sh
```

If you do not work in a VM / do not have root access for any other reason, it is suggested to modify the above build script to use ```singularity build --fakeroot```. Note that this hasn't been tested.
If you want to change the libaries that get installed in the containers modify ``` ai/env-files/pytorch-env-container.txt``` and  ```use-cases/mnist/env-files/preproc-env-container.txt``` to your desire.


### Run the containers

Run preprocessing step with:
```
singularity exec use-cases/mnist/containers/preproc-container.sif python use-cases/mnist/mnist-preproc.py -o use-cases/mnist/data/preproc-images
```

For the AI step the ML server needs to be started first with:
```
singularity exec ai/containers/base-container.sif mlflow server --backend-store-uri use-cases/mnist/data/ml-logs/
```

Then the AI step can be run with:
```
singularity exec ai/containers/base-container.sif itwinai train --train-dataset use-cases/mnist/data/preproc-images --config use-cases/mnist/mnist-ai-train.yml --ml-logs http://127.0.0.1:5000
```

### Future work
- create the containers via GH actions. Building the containers with GH actions works, but uploading the package to the repo fails for some reason.
- run the containers in a workflow with a workflow manager like Apache Airflow, Streamflow, etc.
