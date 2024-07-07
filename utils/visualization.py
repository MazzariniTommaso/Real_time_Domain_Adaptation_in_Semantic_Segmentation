import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_id_to_label
from config import CHECKPOINT_ROOT

# DA RIVEDERE 

def print_stats(epoch:int, 
                train_loss:float,
                val_loss:float, 
                train_miou:float, 
                val_miou:float, 
                verbose:bool)->None:
    """
    Print training and validation statistics if verbose is True.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss value.
        val_loss (float): Validation loss value.
        train_miou (float): Training mean IoU value.
        val_miou (float): Validation mean IoU value.
        verbose (bool): Flag to control verbosity. If False, no output is printed.

    Returns:
        None
    """
    if verbose:
        print(f'Epoch: {epoch}')
        print(f'\tTrain Loss: {train_loss}, Validation Loss: {val_loss}')
        print(f'\tTrain mIoU: {train_miou}, Validation mIoU: {val_miou}')
    
def plot_loss(model_results:list, 
              model_name:str, 
              project_step:str, 
              train_dataset:str, 
              validation_dataset:str)->None:
    """
    Plot and save the training and validation loss curves.

    Args:
        model_results (list): List of model results containing training and validation losses.
        model_name (str): Name of the model.
        project_step (str): Project step or phase.
        train_dataset (str): Name of the training dataset.
        validation_dataset (str): Name of the validation dataset.

    Returns:
        None
    """
    
    epochs = range(len(model_results[0]))
    train_losses = model_results[0]
    validation_losses = model_results[1]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  
    ax.set_title(f'Train vs. Validation Loss for {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.plot(epochs, train_losses, 'o-', color='tab:blue', label=f"train loss ({train_dataset})", linewidth=2, markersize=5)
    ax.plot(epochs, validation_losses, '^-', color='tab:red', label=f"validation loss ({validation_dataset})", linewidth=2, markersize=5)
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()
    
    # Save the plot
    checkpoint_path = f'{CHECKPOINT_ROOT}/{project_step}'
    if os.path.exists(checkpoint_path):
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_loss.png", format='png')
    else:
        os.makedirs(checkpoint_path)
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_loss.png", format='png')

        
def plot_miou(model_results:list, 
              model_name:str, 
              project_step:str, 
              train_dataset:str, 
              validation_dataset:str) -> None:
    """
    Plot and save the training and validation mIoU curves.

    Args:
        model_results (list): List of model results containing training and validation mIoU values.
        model_name (str): Name of the model.
        project_step (str): Project step or phase.
        train_dataset (str): Name of the training dataset.
        validation_dataset (str): Name of the validation dataset.

    Returns:
        None
    """
    epochs = range(len(model_results[2]))
    train_mIoU = model_results[2]
    validation_mIoU = model_results[3]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  
    ax.set_title(f'Train vs. Validation mIoU for {model_name} over Epochs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Mean Intersection over Union (mIoU)', fontsize=14)
    ax.plot(epochs, train_mIoU, 'o-', color='tab:blue', label=f"train mIoU ({train_dataset})", linewidth=2, markersize=5)
    ax.plot(epochs, validation_mIoU, '^-', color='tab:red', label=f"validation mIoU ({validation_dataset})", linewidth=2, markersize=5)
    ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()

    # Save the plot
    checkpoint_path = f'{CHECKPOINT_ROOT}/{project_step}'
    if os.path.exists(checkpoint_path):
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_miou.png", format='png')
    else:
        os.makedirs(checkpoint_path)
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_miou.png", format='png')

def plot_iou(model_results:list, 
              model_name:str, 
              project_step:str, 
              train_dataset:str, 
              validation_dataset:str) -> None:
    """
    Plot and save the IoU (Intersection over Union) for each class across training and validation phases.

    Args:
        model_results (list): List of model results containing training and validation IoU values for each class.
                              It should contain two lists:
                              - model_results[4]: List of training IoU values for each class.
                              - model_results[5]: List of validation IoU values for each class.
        model_name (str): Name of the model.
        project_step (str): Project step or phase.
        train_dataset (str): Name of the training dataset.
        validation_dataset (str): Name of the validation dataset.

    Returns:
        None
    """
    num_classes = 19
    class_names = [get_id_to_label()[i] for i in range(num_classes)]
    train_iou = [model_results[4][i] for i in range(num_classes)]
    val_iou = [model_results[5][i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  
    bar_width = 0.35
    index = np.arange(num_classes)

    ax.bar(index, train_iou, bar_width, label=f'train IoU ({train_dataset})', color='tab:blue', alpha=0.7)
    ax.bar(index + bar_width, val_iou, bar_width, label=f'validation IoU ({validation_dataset})', color='tab:red', alpha=0.7)

    ax.set_xlabel('Classes', fontsize=14)
    ax.set_ylabel('IoU', fontsize=14)
    ax.set_title(f'Training and Validation IoU for Each Class ({model_name})', fontsize=16, fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=12)
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, which='both', linewidth=0.5, axis='y')

    plt.tight_layout()
    plt.show()

    checkpoint_path = f'{CHECKPOINT_ROOT}/{project_step}'
    if os.path.exists(checkpoint_path):
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_iou.png", format='png')
    else:
        os.makedirs(checkpoint_path)
        fig.savefig(f"{checkpoint_path}/{model_name}_{project_step}_iou.png", format='png')
