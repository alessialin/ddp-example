# DDP Example

Submit job on Azure Machine Learning (AML) with the command:

`az ml job create --file aml_job.yaml --resource-group <your-resoure-group> --workspace-name <your-workspace-name>`

Locally:
`torchrun --nnodes=NUM_NODES --nproc_per_node=NUM_GPUS_PER_NODE --node_rank=NODE_INDEX src/train.py`