import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32,64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )

        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifer(x)
        return x

def main():
    parser = argparse.ArgumentParser(description="Train a simple CNN on CIFAR-10")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616),),          
    ])

    train_dataset = datasets.CIFAR10(root=args.data_dir,
                                    train=True,
                                    download=True,
                                    transform=transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir,
                                    train=False,
                                    download=True,
                                    transform=transform)

    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=2,
                            shuffle=True,
                            pin_memory=device.type == "cuda")

    test_loader = DataLoader(test_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=2,
                                    shuffle=False,
                                    pin_memory=device.type == "cuda")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        model.train()
        running_loss=0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct=0
        total=0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                prediction = torch.argmax(outputs, dim = 1)
                correct += (prediction==labels).sum().item()
                total += labels.size(0)

                print(f"Epoch {epoch+1}/{args.epochs}"
                        f"  |Loss:{running_loss}/{len(train_loader):.4f}"
                        f"  |Accuracy:{correct/total*100:.4f}"
                )
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cifar10_cnn.pt"

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()


