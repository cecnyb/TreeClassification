
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split


# Function to plot images before and after normalization
def plot_images_before_after(train_data_unnormalized, train_data_normalized, num_images=5):
    fig, axs = plt.subplots(2, num_images, figsize=(15, 6))
    random_indices = np.random.randint(0, len(train_data_unnormalized), num_images)
    
    for i in range(num_images):
        # Plot unnormalized images
        axs[0, i].imshow(train_data_unnormalized[random_indices[i]].transpose(1, 2, 0))  # Permute to change (C, H, W) to (H, W, C)
        axs[0, i].set_title('Before Normalization')
        axs[0, i].axis('off')
        
        # Plot normalized images
        img = train_data_normalized[random_indices[i]].numpy().transpose(1, 2, 0)  # Permute to change (C, H, W) to (H, W, C)
        axs[1, i].imshow(img)
        axs[1, i].set_title('After Normalization')
        axs[1, i].axis('off')
    
    plt.show()



def train_model(model, train_loader, optimizer, criterion = nn.CrossEntropyLoss(), num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute running loss
            loss_sum += loss.item() * inputs.size(0)

            # Get predictions and compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = loss_sum / total_predictions
        epoch_accuracy = correct_predictions / total_predictions

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

def predict_and_evaluate(model, data_loader):
    model.eval()  
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy


def plot_confusion_matrix(model, data_loader, title, classes):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    #also return the accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {title}')
    plt.show()

    return accuracy


#Define new training loop with early stopping, if the accuracy has not improved for a certain number of epochs (default 3), stop training
def train_model_early_stopping(model, train_loader, val_loader, optimizer, criterion = nn.CrossEntropyLoss(), num_epochs=10, patience=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_val_accuracy = 0
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute running loss
            loss_sum += loss.item() * inputs.size(0)

            # Get predictions and compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = loss_sum / total_predictions
        epoch_accuracy = correct_predictions / total_predictions

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        val_accuracy = predict_and_evaluate(model, val_loader)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


def get_random_test_subset(dataset, subset_size):
    total_size = len(dataset)
    # Create indices for the subset
    indices = torch.randperm(total_size).tolist()
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices), subset_indices


def get_preds(model, data_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    return all_predictions, all_labels