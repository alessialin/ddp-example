# A Distributed Data Parallelization Example
This repository provides an example of Distributed Data Parallelization (DDP) applied to a typical problem of training a CNN on the well-known MNIST dataset. 

## Azure Machine Learning
Submit job on Azure Machine Learning (AML) with the command:

`az ml job create --file aml_job.yaml --resource-group <your-resource-group> --workspace-name <your-workspace-name>`

## Locally
Submit the job locally on your machine:
`python train.py`


## Additional Information
Within the code, we are using `mp.spawn()`. However, if you want to control the number of nodes and processes when launching the script from your command line, then you won't need `mp.spawn()` and use the following command:

`torchrun --nnodes=NUM_NODES --nproc_per_node=NUM_GPUS_PER_NODE --node_rank=NODE_INDEX src/train.py`

Before launching this, you will have to change the `main()` function and run the `train()` function directly.

__NB:__ If you run your training with `mp.spawn`, you cannot launch the script with `torchrun`: they will conflict. Choose one or the other.