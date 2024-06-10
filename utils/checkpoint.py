import torch
from computations import compute_flops, compute_latency_and_fps
from data_processing import get_id_to_label
from typing import List, Tuple
import os

# DA RIVEDERE SAVE_RESULTS

def save_results(model: torch.nn.Module, 
                 model_results: list, 
                 filename: str, 
                 height: int, 
                 width: int, 
                 iterations: int, 
                 ignore_model_measurements: bool = False, 
                 device: str = 'cuda')->None:


    """
    Saves the model results and performance metrics to a specified file.

    Args:
        model (torch.nn.Module): The model whose results are being saved.
        model_results (list): A list containing the model's performance metrics and results.
        filename (str): The name of the file where the results will be saved.
        height (int): The height of the input images.
        width (int): The width of the input images.
        iterations (int): The number of iterations to measure latency and fps.
        ignore_model_measurements (bool, optional): If True, the model measurements will be ignored. Defaults to False.
    This function computes the FLOPS and parameters of the model using the `compute_flops` function,
    and measures the latency and FPS using the `get_latency_and_fps` function. It then writes these
    metrics, along with the training and validation IoU for each class, to a text file in the
    `./results/logs/` directory.
    """


    if not ignore_model_measurements:
        model_params_flops = compute_flops(model, 
                                             height=height, 
                                             width=width)
        model_latency_fps = compute_latency_and_fps(model, 
                                                      height=height,
                                                      width=width,
                                                      iterations=iterations, 
                                                      device = device)
        
    with open(f'./results/logs/{filename}.txt', 'w') as file:
        if not ignore_model_measurements:
            file.write(f"Parameters : {model_params_flops['Parameters']}\n")
            file.write(f"FLOPS : {model_params_flops['FLOPS']}\n")
            file.write(f"Mean Latency = {model_latency_fps[0]}\n")
            file.write(f"STD Latency = {model_latency_fps[1]}\n")
            file.write(f"Mean FPS = {model_latency_fps[2]}\n")
            file.write(f"STD FPS = {model_latency_fps[3]}\n")
        file.write(f'Training Loss = {model_results[0][-1]}\n')
        file.write(f'Validation Loss = {model_results[1][-1]}\n')
        file.write(f'Training mIoU = {model_results[2][-1]}\n')
        file.write(f'Validation mIoU = {model_results[3][-1]}\n')
        for i in range(0, 19):
            file.write(f"Training IoU for class {get_id_to_label()[i]} = {model_results[4][i]}\n")
        for i in range(0, 19):
            file.write(f"Validation IoU for class {get_id_to_label()[i]} = {model_results[5][i]}\n")
            

def save_checkpoint(checkpoint_root: str,
                    step: str, 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    epoch: int,
                    train_loss_list: List[float], 
                    train_miou_list: List[float],
                    train_iou: List[float],
                    val_loss_list: List[float],
                    val_miou_list: List[float],
                    val_iou: List[float]):
    """_summary_

    Args:
        checkpoint_root (str): _description_
        step (str): _description_
        model (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        epoch (int): _description_
        train_loss_list (List[float]): _description_
        train_miou_list (List[float]): _description_
        train_iou (List[float]): _description_
        val_loss_list (List[float]): _description_
        val_miou_list (List[float]): _description_
        val_iou (List[float]): _description_
    """
    
    checkpoint_path = os.path.join(checkpoint_root + "/" + step, "/checkpoint.pth")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'train_loss_list': train_loss_list,
        'train_miou_list': train_miou_list,
        'train_iou': train_iou,
        'val_loss_list': val_loss_list,
        'val_miou_list': val_miou_list,
        'val_iou': val_iou
    }, checkpoint_path)
    print(f"Checkpoint saved in {checkpoint_path} | Epoch: {epoch}")
    
def load_checkpoint(checkpoint_root: str,
                    step: str, 
                    model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer) -> Tuple[int, List[float],List[float],List[float],List[float],List[float],List[float]]:
    """_summary_

    Args:
        checkpoint_root (str): _description_
        step (str): _description_
        model (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_

    Returns:
        Tuple[int, List[float],List[float],List[float],List[float],List[float],List[float]]: _description_
    """
    
    checkpoint_path = os.path.join(checkpoint_root + "/" + step, "/checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        
        checkpoint = torch.load(checkpoint_root + "/" + step + "/checkpoint.pth")
        
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        train_loss_list = checkpoint['train_loss_list']
        train_miou_list = checkpoint['train_miou_list']
        train_iou = checkpoint['train_iou']
        val_loss_list = checkpoint['val_loss_list']
        val_miou_list = checkpoint['val_miou_list']
        val_iou = checkpoint['val_iou']
        
        print(f"Checkpoint found.\tResuming from epoch {start_epoch}.")
        
        return False, start_epoch, train_loss_list, train_miou_list, train_iou, val_loss_list, val_miou_list, val_iou
    
    else:
        dir = checkpoint_root + "/" + step
        os.mkdir(dir)
        print(f"No checkpoint found in {dir}.\tStarting from scratch.")
        return True, None, None, None, None, None, None, None
  