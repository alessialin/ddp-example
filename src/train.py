import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training function
def train(rank, world_size, backend):
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set the device
    torch.manual_seed(0)
    device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )

    # Create the model and move it to the device
    model = SimpleCNN().to(device)
    ddp_model = DDP(
        model, device_ids=[rank] if torch.cuda.is_available() else None
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9)

    # Load the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Training loop
    for epoch in range(5):
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

    # Cleanup
    dist.destroy_process_group()


# Entry point
def main():
    # Check if running in Azure ML
    is_azure_ml = "AZUREML_RUN_ID" in os.environ

    if is_azure_ml:
        # Azure ML environment variables
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        backend = os.environ.get(
            "BACKEND", "nccl" if torch.cuda.is_available() else "gloo"
        )
        train(rank, world_size, backend)
    else:
        # Local execution
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        mp.spawn(
            train,
            args=(world_size, backend),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    main()