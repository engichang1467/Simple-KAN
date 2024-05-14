import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from tqdm import tqdm

from model import KAN


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=False, transform=transform),
    batch_size=64,
    shuffle=False,
)


# Load the saved model and optimizer
model = KAN(input_features=28 * 28, output_features=10, layers_hidden=[64])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# Load the saved model weights and optimizer state
checkpoint = torch.load("model_final.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


for i in range(10):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28)
            output = model(images)
            val_loss += criterion(output, labels).item()
            val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    val_loss /= len(test_loader)
    val_accuracy /= len(test_loader)

    print(f"Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
