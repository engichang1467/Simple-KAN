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

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data", train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True,
)

# Define the model and optimizer
model = KAN(input_features=28 * 28, output_features=10, layers_hidden=[64])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    with tqdm(train_loader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28)
            labels = labels

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            accuracy = (output.argmax(dim=1) == labels).float().mean()

            epoch_loss += loss.item()
            epoch_acc += accuracy.item()

            pbar.set_postfix(
                loss=loss.item(),
                accuracy=accuracy.item(),
                lr=optimizer.param_groups[0]["lr"],
            )

        # Calculate average loss and accuracy for the epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc:.4f}")

    
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28)
            output = model(images)
            val_loss += criterion(output, labels).item()
            val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    
    # Calculate average loss and accuracy for the validation
    val_loss /= len(test_loader)
    val_accuracy /= len(test_loader)

    print(f"Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")


# Save the model
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_final.pt",
)
