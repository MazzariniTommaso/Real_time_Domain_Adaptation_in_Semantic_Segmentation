import torch

def poly_lr_scheduler(optimizer: torch.optim.Optimizer, 
                      init_lr: float, 
                      iter: int, 
                      max_iter:int = 300, 
                      power: float = 0.9) -> float:
    """
    Polynomial learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer object.
        init_lr (float): Initial learning rate.
        iter (int): Current iteration number.
        max_iter (int, optional): Maximum number of iterations (default is 300).
        power (float, optional): Power factor (default is 0.9).

    Returns:
        float: Updated learning rate.
    """

    # Calculate the learning rate using the polynomial decay formula
    lr = init_lr * (1 - iter / max_iter) ** power

    # Update the learning rate in the optimizer
    optimizer.param_groups[0]['lr'] = lr