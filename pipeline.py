import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from tqdm import tqdm
from itertools import cycle
from typing import Tuple, List, Union
from config import CITYSCAPES, GTA, DEEPLABV2_PATH, CITYSCAPES_PATH, GTA5_PATH
from datasets import CityScapes, GTA5
from models import BiSeNet, get_deeplab_v2, FCDiscriminator
from utils import *
import warnings
warnings.filterwarnings("ignore")
torch.cuda.manual_seed(42)


def get_core(model_name: str, 
             n_classes: int,
             device: str,
             parallelize: bool,
             optimizer_name: str, 
             lr: float,
             momentum: float,
             weight_decay: float,
             loss_fn_name: str,
             ignore_index: int,
             adversarial: bool) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module, torch.nn.Module, torch.optim.Optimizer, torch.nn.Module]:
    """
    Set up components for semantic segmentation model training.

    Args:
    - model_name (str): Name of the segmentation model ('DeepLabV2' or 'BiSeNet').
    - n_classes (int): Number of classes in the dataset.
    - device (str): Device to run the model on ('cpu' or 'cuda').
    - parallelize (bool): Whether to use DataParallel for multi-GPU training.
    - optimizer_name (str): Name of the optimizer ('Adam' or 'SGD').
    - lr (float): Learning rate for the optimizer.
    - momentum (float): Momentum factor for SGD optimizer.
    - weight_decay (float): Weight decay (L2 penalty) for the optimizer.
    - loss_fn_name (str): Name of the loss function ('CrossEntropyLoss').
    - ignore_index (int): Index to ignore in loss computation.
    - adversarial (bool): Whether to include adversarial training components.

    Raises:
    - ValueError: If an invalid model_name, optimizer_name, or loss_fn_name is provided.

    Returns:
    - Tuple containing:
        - model (nn.Module): Segmentation model.
        - optimizer (torch.optim.Optimizer): Optimizer for the segmentation model.
        - loss_fn (nn.Module): Loss function for the segmentation model.
        - model_D (nn.Module or None): Discriminator model for adversarial training (if adversarial=True).
        - optimizer_D (torch.optim.Optimizer or None): Optimizer for the discriminator model (if adversarial=True).
        - loss_D (nn.Module or None): Loss function for the discriminator model (if adversarial=True).
    """
    
    model = None
    optimizer = None
    loss_fn = None
    model_D = None
    optimizer_D = None
    loss_D = None
    
    # Initialize segmentation model based on model_name
    if model_name == 'DeepLabV2':
        model = get_deeplab_v2(num_classes=n_classes, pretrain=True, pretrain_model_path=DEEPLABV2_PATH).to(device)
        if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(device)
    elif model_name == 'BiSeNet':
        model = BiSeNet(num_classes=n_classes, context_path="resnet18").to(device)
        if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(device)
    else:
        raise ValueError('Model accepted: [DeepLabV2, BiSeNet]')
            
    # Initialize optimizer based on optimizer_name
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer accepted: [Adam, SGD]')
        
    # Initialize loss function based on loss_fn_name
    if loss_fn_name == 'CrossEntropyLoss':
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        raise ValueError('Loss function accepted: [CrossEntropyLoss]')
    
    # Initialize adversarial components if adversarial is True
    if adversarial:
        model_D = FCDiscriminator(num_classes=n_classes).to(device)
        if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
            model_D = torch.nn.DataParallel(model_D).to(device)
        optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-3, betas=(0.9, 0.99))
        loss_D = torch.nn.BCEWithLogitsLoss()
        
    return model, optimizer, loss_fn, model_D, optimizer_D, loss_D

def get_loaders(train_dataset_name: str, 
                val_dataset_name: str, 
                augmented: bool,
                augmentedType: str,
                batch_size: int,
                n_workers: int,
                adversarial: bool) -> Tuple[Union[DataLoader, Tuple[DataLoader, DataLoader]], DataLoader, int, int]:
    """
    Set up data loaders for training and validation datasets in semantic segmentation.

    Args:
    - train_dataset_name (str): Name of the training dataset ('CityScapes' or 'GTA5').
    - val_dataset_name (str): Name of the validation dataset ('CityScapes').
    - augmented (bool): Whether to use augmented data.
    - augmentedType (str): Type of augmentation to apply (specific to your implementation).
    - batch_size (int): Batch size for data loaders.
    - n_workers (int): Number of workers for data loading.
    - adversarial (bool): Whether to set up adversarial training data loaders.

    Raises:
    - ValueError: If an invalid train_dataset_name or val_dataset_name is provided.

    Returns:
    - Tuple containing:
        - train_loader (Union[DataLoader, Tuple[DataLoader, DataLoader]]): DataLoader(s) for the training dataset.
        - val_loader (DataLoader): DataLoader for the validation dataset.
        - data_height (int): Height of the dataset images.
        - data_width (int): Width of the dataset images.
    """

    transform_cityscapes = A.Compose([
        A.Resize(CITYSCAPES['height'], CITYSCAPES['width']),
    ])
    transform_gta5 = A.Compose([
        A.Resize(GTA['height'], GTA['width'])
    ])

    train_loader = None
    val_loader = None
    data_height = None
    data_width = None
    
    if augmented:
        transform_gta5 = get_augmented_data(augmentedType)
    
    if adversarial:
        source_dataset = GTA5(root_dir=GTA5_PATH, transform=transform_gta5)
        target_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='train', transform=transform_cityscapes)

        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)

        train_loader = (source_loader, target_loader)
    else:
        if train_dataset_name == 'CityScapes':
            train_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='train', transform=transform_cityscapes)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        elif train_dataset_name == 'GTA5':
            train_dataset = GTA5(root_dir=GTA5_PATH, transform=transform_gta5)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        else:
            raise ValueError('Train datasets accepted: [CityScapes, GTA5]')
        
    if val_dataset_name == 'CityScapes':
        val_dataset = CityScapes(root_dir=CITYSCAPES_PATH, split='val', transform=transform_cityscapes)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        data_height = CITYSCAPES['height']
        data_width = CITYSCAPES['width']
    else:
        raise ValueError('Val datasets accepted: [CityScapes]')
    
    return train_loader, val_loader, data_height, data_width

def adversarial_train_step(model: torch.nn.Module, 
                           model_D: torch.nn.Module, 
                           loss_fn: torch.nn.Module, 
                           loss_D: torch.nn.Module, 
                           optimizer: torch.optim.Optimizer, 
                           optimizer_D: torch.optim.Optimizer, 
                           dataloaders: Tuple[DataLoader,DataLoader], 
                           device: str, 
                           n_classes: int = 19)-> Tuple[float, float, float]:
    """
    Perform a single adversarial training step for semantic segmentation.

    Args:
    - model (torch.nn.Module): Segmentation model.
    - model_D (torch.nn.Module): Discriminator model.
    - loss_fn (torch.nn.Module): Segmentation loss function.
    - loss_D (torch.nn.Module): Adversarial loss function for discriminator.
    - optimizer (torch.optim.Optimizer): Optimizer for segmentation model.
    - optimizer_D (torch.optim.Optimizer): Optimizer for discriminator model.
    - dataloaders (Tuple[DataLoader,DataLoader]): Source and target dataloaders for training data.
    - device (str): Device on which to run the models ('cuda' or 'cpu').
    - n_classes (int, optional): Number of classes for segmentation. Default is 19.

    Returns:
    - Tuple containing:
        - epoch_loss (float): Average segmentation loss for the epoch.
        - epoch_miou (float): Mean Intersection over Union (mIoU) for the epoch.
        - epoch_iou (np.ndarray): Array of per-class IoU values for the epoch.
    """

    model_G = model.to(device)
    optimizer_G = optimizer
    ce_loss = loss_fn
    bce_loss = loss_D
    
    interp_source = nn.Upsample(size=(GTA['height'], GTA['width']), mode='bilinear')
    interp_target = nn.Upsample(size=(CITYSCAPES['height'], CITYSCAPES['width']), mode='bilinear')
    
    lambda_adv = 0.001
    total_loss = 0
    total_miou = 0
    total_iou = np.zeros(n_classes)
    
    iterations = 0
    
    model_G.train()
    model_D.train()
    
    source_loader, target_loader = dataloaders
    train_loader = zip(source_loader, cycle(target_loader))
    
    
    for (source_data, source_labels), (target_data, _) in train_loader:
        
        iterations+=1

        source_data, source_labels = source_data.to(device), source_labels.to(device)
        target_data = target_data.to(device)
        
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        #TRAIN GENERATOR
        
        #Train with source
        for param in model_D.parameters():
            param.requires_grad = False
        
        output_source = model_G(source_data)
        output_source = interp_source(output_source) # apply upsample

        segmentation_loss = ce_loss(output_source, source_labels)
        segmentation_loss.backward()

        #Train with target
        output_target = model_G(target_data)
        output_target = interp_target(output_target) # apply upsample
        
        prediction_target = torch.nn.functional.softmax(output_target)
        discriminator_output_target = model_D(prediction_target)
        discriminator_label_source = torch.FloatTensor(discriminator_output_target.data.size()).fill_(0).cuda() # 0 = source domain
        
        adversarial_loss = bce_loss(discriminator_output_target, discriminator_label_source)
        discriminator_loss = lambda_adv * adversarial_loss
        discriminator_loss.backward()
        
        
        #TRAIN DISCRIMINATOR
        
        #Train with source
        for param in model_D.parameters():
            param.requires_grad = True
            
        output_source = output_source.detach()
        
        prediction_source = torch.nn.functional.softmax(output_source)
        discriminator_output_source = model_D(prediction_source)
        discriminator_label_source = torch.FloatTensor(discriminator_output_source.data.size()).fill_(0).cuda() # 0 = source domain
        discriminator_loss_source = bce_loss(discriminator_output_source, discriminator_label_source)
        discriminator_loss_source.backward()

        #Train with target
        output_target = output_target.detach()
        
        prediction_target = torch.nn.functional.softmax(output_target)
        discriminator_output_target = model_D(prediction_target)
        discriminator_label_target = torch.FloatTensor(discriminator_output_target.data.size()).fill_(1).cuda() # 1 = target domain
        
        discriminator_loss_target = bce_loss(discriminator_output_target, discriminator_label_target)
        discriminator_loss_target.backward()
        
        optimizer_G.step()
        optimizer_D.step()
        
        total_loss += segmentation_loss.item()
        
        prediction_source = torch.argmax(torch.softmax(output_source, dim=1), dim=1)
        hist = fast_hist(source_labels.cpu().numpy(), prediction_source.cpu().numpy(), n_classes)
        running_iou = np.array(per_class_iou(hist)).flatten()
        total_miou += running_iou.sum()
        total_iou += running_iou

        
    epoch_loss = total_loss / iterations
    epoch_miou = total_miou / (iterations * n_classes)
    epoch_iou = total_iou / iterations
    
    return epoch_loss, epoch_miou, epoch_iou

def train_step(model: torch.nn.Module, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               dataloader: DataLoader, 
               device: str, 
               n_classes: int = 19)-> Tuple[float, float, float]:
    """
    Perform a single training step for semantic segmentation.

    Args:
    - model (torch.nn.Module): Segmentation model.
    - loss_fn (torch.nn.Module): Loss function for segmentation.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - dataloader (DataLoader): DataLoader for training data.
    - device (str): Device on which to run the models ('cuda' or 'cpu').
    - n_classes (int, optional): Number of classes for segmentation. Default is 19.

    Returns:
    - Tuple containing:
        - epoch_loss (float): Average segmentation loss for the epoch.
        - epoch_miou (float): Mean Intersection over Union (mIoU) for the epoch.
        - epoch_iou (np.ndarray): Array of per-class IoU values for the epoch.
    """

    total_loss = 0
    total_miou = 0
    total_iou = np.zeros(n_classes)
    
    model.train()
    
    for image, label in dataloader:
        image, label = image.to(device), label.type(torch.LongTensor).to(device)
    
        output = model(image)
        loss = loss_fn(output, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)

        hist = fast_hist(label.cpu().numpy(), prediction.cpu().numpy(), n_classes)
        running_iou = np.array(per_class_iou(hist)).flatten()
        total_miou += running_iou.sum()
        total_iou += running_iou
    
    epoch_loss = total_loss / len(dataloader)
    epoch_miou = total_miou / (len(dataloader)* n_classes)
    epoch_iou = total_iou / len(dataloader)
    
    return epoch_loss, epoch_miou, epoch_iou

def val_step(model: torch.nn.Module,  
             loss_fn: torch.nn.Module, 
             dataloader: DataLoader, 
             device: str, 
             n_classes: int = 19) -> Tuple[float, float, float]:
    """
    Perform a single validation step for semantic segmentation.

    Args:
    - model (torch.nn.Module): Segmentation model.
    - loss_fn (torch.nn.Module): Loss function for segmentation.
    - dataloader (DataLoader): DataLoader for validation data.
    - device (str): Device on which to run the models ('cuda' or 'cpu').
    - n_classes (int, optional): Number of classes for segmentation. Default is 19.

    Returns:
    - Tuple containing:
        - epoch_loss (float): Average segmentation loss for the epoch.
        - epoch_miou (float): Mean Intersection over Union (mIoU) for the epoch.
        - epoch_iou (np.ndarray): Array of per-class IoU values for the epoch.
    """
    
    total_loss = 0
    total_miou = 0
    total_iou = np.zeros(n_classes)
    
    model.eval()

    with torch.inference_mode(): # which is analogous to torch.no_grad
        for image, label in dataloader:
            image, label = image.to(device), label.type(torch.LongTensor).to(device)
            
            output = model(image)
            loss = loss_fn(output, label)
            total_loss += loss.item()
            
            prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
            
            hist = fast_hist(label.cpu().numpy(), prediction.cpu().numpy(), n_classes)
            running_iou = np.array(per_class_iou(hist)).flatten()
            total_miou += running_iou.sum()
            total_iou += running_iou
    
    epoch_loss = total_loss / len(dataloader)
    epoch_miou = total_miou / (len(dataloader)* n_classes)
    epoch_iou = total_iou / len(dataloader)
    
    return epoch_loss, epoch_miou, epoch_iou
    
def train(model: torch.nn.Module, 
          model_D: torch.nn.Module, 
          optimizer: torch.optim.Optimizer, 
          optimizer_D: torch.optim.Optimizer, 
          loss_fn: torch.nn.Module, 
          loss_D: torch.nn.Module, 
          train_loader: Union[DataLoader, Tuple[DataLoader,DataLoader]],  
          val_loader: DataLoader, 
          epochs: int, 
          device: str, 
          checkpoint_root: str,
          project_step: str,
          verbose: bool,
          n_classes: int = 19,
          power: float = 0.9,
          adversarial: bool = False) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Train a semantic segmentation model with optional adversarial training.

    Args:
        model (torch.nn.Module): Semantic segmentation model.
        model_D (torch.nn.Module): Discriminator model for adversarial training.
        optimizer (torch.optim.Optimizer): Optimizer for the segmentation model.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        loss_fn (torch.nn.Module): Loss function for segmentation.
        loss_D (torch.nn.Module): Loss function for adversarial training.
        train_loader (Union[DataLoader, Tuple[DataLoader,DataLoader]]): DataLoader(s) for the training dataset.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of epochs to train.
        device (str): Device on which to run computations ('cuda' or 'cpu').
        checkpoint_root (str): Root directory to save checkpoints.
        project_step (str): Name/id of the project or step.
        verbose (bool): Whether to print verbose training statistics.
        n_classes (int, optional): Number of classes for segmentation. Defaults to 19.
        power (float, optional): Power parameter for learning rate scheduler. Defaults to 0.9.
        adversarial (bool, optional): Whether to use adversarial training. Defaults to False.

    Returns:
        Tuple containing lists of:
        - train_loss_list (List[float]): List of training losses per epoch.
        - val_loss_list (List[float]): List of validation losses per epoch.
        - train_miou_list (List[float]): List of training mIoU per epoch.
        - val_miou_list (List[float]): List of validation mIoU per epoch.
        - train_iou (List[float]): List of per-class IoU for training per epoch.
        - val_iou (List[float]): List of per-class IoU for validation per epoch.
    """
    
    # Load or initialize checkpoint
    no_checkpoint, start_epoch, train_loss_list, train_miou_list, train_iou, val_loss_list, val_miou_list, val_iou = load_checkpoint(checkpoint_root=checkpoint_root, project_step=project_step, adversarial=adversarial, model=model, model_D=model_D, optimizer=optimizer, optimizer_D=optimizer_D)
        
    if no_checkpoint:
        train_loss_list, train_miou_list = [], []
        val_loss_list, val_miou_list = [], []
        start_epoch = 0
    
    for epoch in tqdm(range(start_epoch, epochs)):
        
        # Perform training step
        if adversarial:
            train_loss, train_miou, train_iou = adversarial_train_step(model=model,
                                                                       model_D=model_D,
                                                                       loss_fn=loss_fn, 
                                                                       loss_D=loss_D, 
                                                                       optimizer=optimizer, 
                                                                       optimizer_D=optimizer_D, 
                                                                       dataloaders=train_loader, 
                                                                       device=device, 
                                                                       n_classes=n_classes)
        else:
            train_loss, train_miou, train_iou = train_step(model=model, 
                                                           loss_fn=loss_fn, 
                                                           optimizer=optimizer, 
                                                           dataloader=train_loader, 
                                                           device=device, 
                                                           n_classes=n_classes)
        
        # Perform validation step
        val_loss, val_miou, val_iou = val_step(model=model, 
                                               loss_fn=loss_fn, 
                                               dataloader=val_loader,
                                               device=device, 
                                               n_classes=n_classes)
        
        # Append metrics to lists
        train_loss_list.append(train_loss) 
        train_miou_list.append(train_miou) 
        val_loss_list.append(val_loss)
        val_miou_list.append(val_miou)

        # Print statistics if verbose
        print_stats(epoch=epoch, 
                    train_loss=train_loss,
                    val_loss=val_loss, 
                    train_miou=train_miou, 
                    val_miou=val_miou, 
                    verbose=verbose)

        # Adjust learning rate
        poly_lr_scheduler(optimizer=optimizer,
                          init_lr=optimizer.param_groups[0]['lr'],
                          iter=epoch, 
                          max_iter=epochs,
                          power=power)
        if adversarial:
            poly_lr_scheduler(optimizer=optimizer_D,
                              init_lr=optimizer_D.param_groups[0]['lr'],
                              iter=epoch, 
                              max_iter=epochs,
                              power=power)
        
        # Save checkpoint after each epoch
        save_checkpoint(checkpoint_root=checkpoint_root, 
                        project_step=project_step,
                        adversarial=adversarial,
                        model=model, 
                        model_D=model_D,
                        optimizer=optimizer, 
                        optimizer_D=optimizer_D, 
                        epoch=epoch,
                        train_loss_list=train_loss_list, 
                        train_miou_list=train_miou_list,
                        train_iou=train_iou,
                        val_loss_list=val_loss_list,
                        val_miou_list=val_miou_list,
                        val_iou=val_iou,
                        verbose=verbose)
        
    return train_loss_list, val_loss_list, train_miou_list, val_miou_list, train_iou, val_iou
    
def pipeline (model_name: str, 
              train_dataset_name: str, 
              val_dataset_name: str,
              n_classes:int,
              epochs: int,
              augmented: bool,
              augmentedType:str,
              optimizer_name: str,
              lr:float,
              momentum:float,
              weight_decay:float,
              loss_fn_name: str,
              ignore_index:int,
              batch_size: int,
              n_workers: int,
              device:str,
              parallelize:bool,
              project_step:str,
              verbose: bool,
              checkpoint_root:str,
              power:float,
              evalIterations:int,
              adversarial:bool
              )->None:
    """
    Main pipeline function to orchestrate the training and evaluation of a deep learning model.

    Args:
        model_name (str): Name of the deep learning model architecture.
        train_dataset_name (str): Name of the training dataset.
        val_dataset_name (str): Name of the validation dataset.
        n_classes (int): Number of classes in the dataset.
        epochs (int): Number of epochs for training.
        augmented (bool): Whether to use data augmentation during training.
        augmentedType (str): Type of data augmentation to apply.
        optimizer_name (str): Name of the optimizer to use.
        lr (float): Learning rate for the optimizer.
        momentum (float): Momentum factor for optimizers like SGD.
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        loss_fn_name (str): Name of the loss function.
        ignore_index (int): Index to ignore in the loss function (e.g., for padding).
        batch_size (int): Batch size for training and validation data loaders.
        n_workers (int): Number of workers for data loading.
        device (str): Device to run the model on ('cuda' or 'cpu').
        parallelize (bool): Whether to use GPU parallelization.
        project_step (str): Name or identifier of the current project step or experiment.
        verbose (bool): Whether to print detailed logs during training.
        checkpoint_root (str): Root directory to save checkpoints and results.
        power (float): Power parameter for polynomial learning rate scheduler.
        evalIterations (int): Number of iterations for evaluating model latency and FPS.
        adversarial (bool): Whether to use adversarial training.

    Returns:
        None
    """
    
    
    # get model
    model, optimizer, loss_fn, model_D, optimizer_D, loss_D = get_core(model_name, 
                                                                       n_classes,
                                                                       device,
                                                                       parallelize,
                                                                       optimizer_name, 
                                                                       lr,
                                                                       momentum,
                                                                       weight_decay,
                                                                       loss_fn_name,
                                                                       ignore_index,
                                                                       adversarial)
    # get loader
    train_loader, val_loader, data_height, data_width = get_loaders(train_dataset_name, 
                                                                    val_dataset_name, 
                                                                    augmented,
                                                                    augmentedType,
                                                                    batch_size,
                                                                    n_workers,
                                                                    adversarial)
    # train
    model_results = train(model=model,
                          model_D = model_D,
                          optimizer=optimizer, 
                          optimizer_D = optimizer_D,
                          loss_fn = loss_fn, 
                          loss_D = loss_D,
                          train_loader=train_loader, 
                          val_loader=val_loader, 
                          epochs=epochs, 
                          device=device, 
                          checkpoint_root=checkpoint_root,
                          project_step=project_step,
                          verbose=verbose,
                          n_classes=n_classes,
                          power=power,
                          adversarial=adversarial)
    
    # evaluation
    model_params_flops = compute_flops(model=model, 
                                       height=data_height, 
                                       width=data_width)
    
    model_latency_fps = compute_latency_and_fps(model=model,
                                                height=data_height, 
                                                width=data_width, 
                                                iterations=evalIterations, 
                                                device=device)
    
    # visualization
    plot_loss(model_results, 
              model_name, 
              project_step, 
              train_dataset_name, 
              val_dataset_name)
    
    plot_miou(model_results, 
              model_name, 
              project_step, 
              train_dataset_name, 
              val_dataset_name)
    
    plot_iou(model_results, 
             model_name, 
             project_step, 
             train_dataset_name, 
             val_dataset_name)
    
    # save results
    save_results(model_results, 
                 filename=f"{model_name}_metrics_{project_step}", 
                 project_step=project_step,
                 model_params_flops=model_params_flops,
                 model_latency_fps=model_latency_fps)