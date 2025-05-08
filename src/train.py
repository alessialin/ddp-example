import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST digit classification.
    Architecture:
        - 2 convolutional layers with ReLU and max pooling
        - 2 fully connected layers
    Input shape: (batch_size, 1, 28, 28)
    Output shape: (batch_size, 10)
    """
    def __init__(self) -> None:
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def setup(backend: str) -> None:
    """
    Setup the distributed environment.

    Args:
        backend (str): Backend to use for distributed training
            ("nccl" or "gloo").

    """

    dist.init_process_group(
        backend=backend,
        init_method="env://"
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return device


# Training function
def train(rank: int, world_size: int, backend: str, epochs: int = 5) -> None:
    """
    Distributed training loop for MNIST using PyTorch DDP.

    Args:
        rank (int): The rank of the current process in the distributed setup.
        world_size (int): Total number of processes (GPUs across all nodes).
        backend (str): Backend to use for distributed training
            ("nccl" or "gloo").
        epochs (int): Number of training epochs.
    """
    # Initialize the process group (if gpu init_method is env:// by default)
    device = setup(backend)

    # Set the device
    torch.manual_seed(42)

    # Create the model and move it to the device
    model = SimpleCNN().to(device)
    ddp_model = DDP(model)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)

    # Load the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Save the model
    if rank == 0:
        model_dir = "./outputs"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "mnist_cnn.pt")
        # Save the underlying model (not the DDP wrapper)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Cleanup
    dist.destroy_process_group()


def local_run(rank: int, world_size: int, backend: str) -> None:
    """
    Setup the distributed environment.

    Args:
        rank (int): The rank of the current process in the distributed setup.
        world_size (int): Total number of processes (GPUs across all nodes).
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    train(rank, world_size, backend)


def main() -> None:
    """
    Entry point for the script. Detects environment (Azure ML or local),
    sets up distributed training, and launches the training function.
    """
    # Check if running in Azure ML
    is_azure_ml = "AZUREML_RUN_ID" in os.environ

    if is_azure_ml:
        # Azure ML environment variables
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        backend = os.environ.get(
            "BACKEND", "nccl" if torch.cuda.is_available() else "gloo"
        )
        print(
            f"Running in Azure ML with {world_size} processes using {backend}"
        )
        train(rank, world_size, backend)
    else:
        # Local execution
        world_size = (
            torch.cuda.device_count() if torch.cuda.is_available() else 2
        )
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        print(
            f"Running locally with {world_size} processes using {backend}"
        )

        # Spawn processes for distributed training
        mp.spawn(
            local_run,
            args=(world_size, backend),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()
