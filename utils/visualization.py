import matplotlib.pyplot as plt
from utils import get_id_to_label
import numpy as np

# DA RIVEDERE 

def print_stats(epoch:int, 
                train_loss:float,
                val_loss:float, 
                train_miou:float, 
                val_miou:float, 
                verbose:bool):
    
    """_summary_

    Args:
        epoch (int): _description_
        train_loss (float): _description_
        val_loss (float): _description_
        train_miou (float): _description_
        val_miou (float): _description_
        verbose (int): _description_

    Returns:
        _type_: _description_
    """
    
    if verbose == False:
        return 0
    elif verbose == True:
        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss}, Validation Loss: {val_loss}')
        print(f'Train mIoU: {train_miou}, Validation mIoU: {val_miou}')
        print('-'*20)
        return 0
    
def plot_loss(model_results:list, model_name:str, phase:str, train_dataset:str, validation_dataset:str):
    """
    Plots the training and validation loss over epochs for a given model, optimized for academic publications.

    Parameters:
        model_results (list): A list containing the loss values over epochs for both training and validation.
        model_name (str): The name of the model being evaluated.
        phase (str): The current phase in the training/validation process.
        train_dataset (str): The name of the training dataset.
        validation_dataset (str): The name of the validation dataset.
    """
    epochs = range(len(model_results[0]))
    train_losses = model_results[0]
    validation_losses = model_results[1]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_title(f'Train vs. Validation Loss for {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.plot(epochs, train_losses, 'o-', label=f"Train Loss - {train_dataset}", linewidth=2, markersize=5)
    ax.plot(epochs, validation_losses, 's--', label=f"Validation Loss - {validation_dataset}", linewidth=2, markersize=5)
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()
    fig.savefig(f"./results/images/{model_name}_{phase}_loss_high_res.svg", format='svg')

def plot_miou(model_results:list, model_name:str, phase:str, train_dataset:str, validation_dataset:str):
    """
    Plots the training and validation mean Intersection over Union (mIoU) over epochs for a given model, optimized for academic publications.

    Parameters:
        model_results (list): A list containing the mIoU values over epochs for both training and validation.
        model_name (str): The name of the model being evaluated.
        phase (str): The current phase in the training/validation process.
        train_dataset (str): The name of the training dataset.
        validation_dataset (str): The name of the validation dataset.
    """
    epochs = range(len(model_results[2]))
    train_mIoU = model_results[2]
    validation_mIoU = model_results[3]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_title(f'Train vs. Validation mIoU for {model_name} over Epochs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Mean Intersection over Union (mIoU)', fontsize=14)
    ax.plot(epochs, train_mIoU, 'o-', label=f"Train mIoU - {train_dataset}", linewidth=2, markersize=5)
    ax.plot(epochs, validation_mIoU, 's--', label=f"Validation mIoU - {validation_dataset}", linewidth=2, markersize=5)
    ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()
    fig.savefig(f"./results/images/{model_name}_{phase}_mIoU_academic.svg", format='svg')

def plot_iou(model_results:list, model_name:str, phase:str, train_dataset:str, validation_dataset:str):
    """
    Plots the training and validation Intersection over Union (IoU) for each class over epochs for a given model, optimized for academic publications.

    Parameters:
        model_results (list): A list containing the IoU values for each class over epochs for both training and validation.
        model_name (str): The name of the model being evaluated.
        phase (str): The current phase in the training/validation process.
        train_dataset (str): The name of the training dataset.
        validation_dataset (str): The name of the validation dataset.
    """
    num_classes = 19
    class_names = [get_id_to_label()[i] for i in range(num_classes)]
    train_iou = [model_results[4][i] for i in range(num_classes)]
    val_iou = [model_results[5][i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    bar_width = 0.35
    index = np.arange(num_classes)

    ax.bar(index, train_iou, bar_width, label=f'Train IoU - {train_dataset}', color='blue', alpha=0.7)
    ax.bar(index + bar_width, val_iou, bar_width, label=f'Validation IoU - {validation_dataset}', color='green', alpha=0.7)

    ax.set_xlabel('Classes', fontsize=14)
    ax.set_ylabel('IoU', fontsize=14)
    ax.set_title(f'Training and Validation IoU for Each Class ({model_name})', fontsize=16, fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

    plt.tight_layout()
    plt.show()
    fig.savefig(f"./results/images/{model_name}_{phase}_IoU_barplot_academic.svg", format='svg')
