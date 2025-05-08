# A Distributed Data Parallel Example with PyTorch
This repository provides an example of Distributed Data Parallel (DDP) applied to a typical problem of training a CNN on the well-known MNIST dataset. 

## Prerequisites
- Azure Machine Learning (AML) subscription
- AML workspace
- GPU quota
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
- Python environment (3.12) with libraries within `requirements.txt`

## Repo Structure
```
ddp-example/
├── src/
│   └── aml_job.yaml       # Script to run job on aml
│   └── train.py           # Training script
├── [output and data folders]       # gitignored
```

## Running on Azure Machine Learning
Submit job on Azure Machine Learning (AML) with the command:

`az ml job create --file aml_job.yaml --resource-group <your-resource-group> --workspace-name <your-workspace-name>`

## Running Locally
Submit the job locally on your machine:
`python train.py`


## Additional Information
Within the code, we are using `mp.spawn()`. However, if you want to control the number of nodes and processes when launching the script from your command line, then you won't need `mp.spawn()` and use the following command:

`torchrun --nnodes=NUM_NODES --nproc_per_node=NUM_GPUS_PER_NODE --node_rank=NODE_INDEX src/train.py`

Before launching this, you will have to change the `main()` function and run the `train()` function directly.

__NB:__ If you run your training with `mp.spawn`, you cannot launch the script with `torchrun`: they will conflict. Choose one or the other.