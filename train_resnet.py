import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from codecarbon import EmissionsTracker
import time
import json
from datetime import datetime
import platform
import os

def get_gpu_name():
    """Get GPU model name"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU"

def train_model(cloud_provider, region, instance_type):
    # Metadata collection
    metadata = {
        "cloud_provider": cloud_provider,
        "region": region,
        "instance_type": instance_type,
        "gpu_model": get_gpu_name(),
        "start_time": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
    }
    
    print(f"\n{'='*50}")
    print(f"Starting experiment on {cloud_provider}")
    print(f"Region: {region}")
    print(f"Instance: {instance_type}")
    print(f"GPU: {metadata['gpu_model']}")
    print(f"{'='*50}\n")
    
    # Start CodeCarbon tracking
    tracker = EmissionsTracker(
        project_name=f"{cloud_provider}_resnet_training",
        output_dir="./results",
        output_file=f"{cloud_provider.lower()}_{region}_emissions.csv",
    )
    tracker.start()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    
    # CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    # Load ResNet18
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss:.4f}')
    
    training_time = time.time() - start_time
    
    # Stop tracking
    emissions_data = tracker.stop()
    
    # Complete metadata
    metadata["end_time"] = datetime.now().isoformat()
    metadata["runtime_seconds"] = training_time
    metadata["runtime_minutes"] = training_time / 60
    metadata["num_epochs"] = num_epochs
    metadata["batch_size"] = batch_size
    metadata["final_loss"] = epoch_losses[-1]
    metadata["energy_kwh"] = emissions_data  # emissions in kg
    metadata["notes"] = "Training completed successfully"
    
    # Save metadata to JSON
    os.makedirs("./results", exist_ok=True)
    metadata_file = f"./results/{cloud_provider.lower()}_{region}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Runtime: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Metadata saved to: {metadata_file}")
    print(f"Emissions data saved to: ./results/{cloud_provider.lower()}_{region}_emissions.csv")
    print(f"{'='*50}\n")
    
    return metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-10')
    parser.add_argument('--provider', type=str, required=True, 
                       choices=['AWS', 'GCP', 'Azure'],
                       help='Cloud provider name')
    parser.add_argument('--region', type=str, required=True,
                       help='Cloud region')
    parser.add_argument('--instance', type=str, required=True,
                       help='Instance type')
    
    args = parser.parse_args()
    
    train_model(args.provider, args.region, args.instance)
