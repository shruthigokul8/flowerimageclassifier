import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torchvision.models as models
import torch.nn.functional as F
import json
import argparse

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("data_dir", type=str, help="Dataset directory")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", help="Pre-trained model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')    
    return parser.parse_args()

def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Set up the data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    # Dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'train_loader': torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True)

    }

    # Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Load pre-trained model
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define the classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    # Set up loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Move to GPU if available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Validation step
        model.eval()
        accuracy = 0
        valid_loss = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        model.train()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
              f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

    # Save checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'model': arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs
    }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')

if __name__ == "__main__":
    args = get_input_args()
    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
