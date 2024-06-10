import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time

def compute_flops(model:torch.nn.Module, 
                  height:int = 512, 
                  width:int = 1024)->dict:


    """
    Computes the number of floating point operations (FLOPS) for a given model.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        height (int): The height of the input images.
        width (int): The width of the input images.
    Returns:
        str: A string containing the FLOPS count for the model.
    """


    image = torch.zeros((1,3, height, width))

    flops = FlopCountAnalysis(model.cpu(), image)
    table = flop_count_table(flops)
    
    # Extraxt values from the table
    n_param_table = table.split('\n')[2].split('|')[2:][0].strip()
    flops_table = table.split('\n')[2].split('|')[2:][1].strip()

    return {'Parameters': n_param_table,
            'FLOPS': flops_table
            }

def compute_latency_and_fps(model: torch.nn.Module, 
                        height: int = 512, 
                        width: int = 1024, 
                        iterations: int = 1000, 
                        device: str = 'cuda') -> tuple:


    """
    Measures the latency and frames per second (FPS) of a given model on a specified input size over a number of iterations.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        height (int): The height of the input images.
        width (int): The width of the input images.
        iterations (int, optional): The number of iterations to measure. Defaults to 1000.

    Returns:
        tuple: A tuple containing the mean latency in milliseconds, the standard deviation of latency in milliseconds,
               the mean FPS, and the standard deviation of FPS.
    """


    latencies = []
    fps_records = []
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for _ in range(iterations):
            image = torch.zeros((1, 3, height, width)).to(device)
            
            start_time = time.time()
            model(image)
            end_time = time.time() 
            
            latency = end_time - start_time
            latencies.append(latency)
            fps_records.append(1 / latency)
    
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    mean_fps = np.mean(fps_records)
    std_fps = np.std(fps_records)
    
    return mean_latency, std_latency, mean_fps, std_fps

