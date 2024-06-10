import numpy as np
import torch
from typing import Tuple, List
from utils import fast_hist, per_class_iou, load_checkpoint, print_stats, poly_lr_scheduler, save_checkpoint
from tqdm import tqdm
from models import BiSeNet, get_deeplab_v2
from datasets import CityScapes, GTA5
from torch.utils.data import DataLoader
from config import CITYSCAPES

# Training step
def train_step(model: torch.nn.Module, 
               model_name: str, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               dataloader: torch.utils.data.DataLoader, 
               device: str, 
               n_classes: int = 19)-> Tuple[float, float, float]:
    
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        model_name (str): _description_
        loss_fn (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        dataloader (torch.utils.data.DataLoader): _description_
        device (str): _description_
        n_classes (int, optional): _description_. Defaults to 19.

    Returns:
        Tuple[float, float, float]: _description_
    """


    total_loss = 0
    total_miou = 0
    total_iou = np.zeros(n_classes)
    model.train()
    
    for image, label in dataloader:
        image, label = image.to(device), label.type(torch.LongTensor).to(device)
    
        if model_name == 'bisenet':
            output, sup1, sup2 = model(image) # DA CONTROLLARE
            print(output.size())
            print(output)
            loss = loss_fn(output, label)
        else: 
            output = model(image)
            loss = loss_fn(output, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        _, predicted = output[0].max(1) # DA CONTROLLARE
        print(predicted)
        print(predicted.size())
        
        hist = fast_hist(predicted.cpu().numpy(), label.cpu().numpy(), n_classes)
        running_iou = np.array(per_class_iou(hist)).flatten()
        total_miou += running_iou.sum()
        total_iou += running_iou
    
    epoch_loss = total_loss / len(dataloader)
    epoch_miou = total_miou / (len(dataloader)* n_classes)
    epoch_iou = total_iou / len(dataloader)
    
    return epoch_loss, epoch_miou, epoch_iou

# Validation step
def val_step(model: torch.nn.Module,  
             loss_fn: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             device: str, 
             n_classes: int = 19) -> Tuple[float, float, float]:
    
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        loss_fn (torch.nn.Module): _description_
        dataloader (torch.utils.data.DataLoader): _description_
        device (str): _description_
        n_classes (int, optional): _description_. Defaults to 19.

    Returns:
        Tuple[float, float, float]: _description_
    """
    
    total_loss = 0
    total_miou = 0
    total_iou = np.zeros(n_classes)
    model.eval()

    with torch.inference_mode(): # which is analogous to torch.no_grad
        for image, label in enumerate(dataloader):
            image, label = image.cuda(), label.type(torch.LongTensor).to(device)
            
            output = model(image)
            loss = loss_fn(output, label)
            total_loss += loss.item()
            
            _, predicted = output[0].max(1) # DA CONTROLLARE
            
            hist = fast_hist(predicted.cpu().numpy(), label.cpu().numpy(), n_classes)
            running_iou = np.array(per_class_iou(hist)).flatten()
            total_miou += running_iou.sum()
            total_iou += running_iou
    
    epoch_loss = total_loss / len(dataloader)
    epoch_miou = total_miou / (len(dataloader)* n_classes)
    epoch_iou = total_iou / len(dataloader)
    
    return epoch_loss, epoch_miou, epoch_iou
    
# Training/Validation Loop
def train(model: torch.nn.Module, 
          model_name: str, 
          optimizer: torch.optim.Optimizer, 
          loss_fn: torch.nn.Module, 
          train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, 
          epochs: int, 
          device: str, 
          checkpoint_root: str,
          step: str,
          verbose: bool,
          n_classes: int = 19,
          power: float = 0.9) -> Tuple[List[float],List[float],List[float],List[float],List[float],List[float]]:
    
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        model_name (str): _description_
        optimizer (torch.optim.Optimizer): _description_
        loss_fn (torch.nn.Module): _description_
        train_loader (torch.utils.data.DataLoader): _description_
        val_loader (torch.utils.data.DataLoader): _description_
        epochs (int): _description_
        device (str): _description_
        checkpoint_root (str): _description_
        step (str): _description_
        verbose (int, optional): _description_. Defaults to 0.
        n_classes (int, optional): _description_. Defaults to 19.
        power (float, optional): _description_. Defaults to 0.9.

    Returns:
        Tuple[List[float],List[float],List[float],List[float],List[float],List[float]]: _description_
    """
    
    
    # CHECK THE INPUT VALUE BEFORE
    
    # Load last checkpoint
    no_loading, start_epoch, train_loss_list, train_miou_list, train_iou, val_loss_list, val_miou_list, val_iou = load_checkpoint(checkpoint_root=checkpoint_root, step=step, model=model, optimizer=optimizer)
        
    init_lr = optimizer.param_groups[0]['lr']
    
    if no_loading:
        train_loss_list, train_miou_list = [], []
        val_loss_list, val_miou_list = [], []
        init_lr = optimizer.param_groups[0]['lr']
        start_epoch = 0
    
    for epoch in tqdm(range(start_epoch, epochs)):
        
        train_loss, train_miou, train_iou = train_step(model, 
                                                       model_name, 
                                                       loss_fn, 
                                                       optimizer, 
                                                       train_loader, 
                                                       device, 
                                                       n_classes)
        
        val_loss, val_miou, val_iou = val_step(model, 
                                               loss_fn, 
                                               val_loader,
                                               device, 
                                               n_classes)
        
        train_loss_list.append(train_loss) 
        train_miou_list.append(train_miou) 
        val_loss_list.append(val_loss)
        val_miou_list.append(val_miou)

        print_stats(epoch=epoch, 
                    train_loss=train_loss,
                    val_loss=val_loss, 
                    train_miou=train_miou, 
                    val_miou=val_miou, 
                    verbose=verbose)

        poly_lr_scheduler(optimizer = optimizer,
                          init_lr = init_lr,
                          iter = epoch, 
                          max_iter = epochs,
                          power = power)
        
        # Save last checkpoint
        save_checkpoint(checkpoint_root = checkpoint_root, 
                        step=step,
                        model = model, 
                        optimizer = optimizer, 
                        epoch = epoch,
                        train_loss_list = train_loss_list, 
                        train_miou_list = train_miou_list,
                        train_iou = train_iou,
                        val_loss_list = val_loss_list,
                        val_miou_list = val_miou_list,
                        val_iou = val_iou)
        
    return train_loss_list, train_miou_list, train_iou, val_loss_list, val_miou_list, val_iou

def get_data(train=True):
    if train == True:
        # train dataset
        dataset = cityscapes.CityScapes(root_dir=root_dir, 
                                        split='train', 
                                        image_transform=image_transform, 
                                        label_transform=label_transform)
    else:
        # test dataset
        dataset = CityScapes(root_dir=root_dir, 
                             split='val', 
                             image_transform=image_transform, 
                             label_transform=label_transform)
    
    return dataset

def make_loader(dataset, batch_size=4, train=True, num_workers=4):
    if train:
        # train dataloader
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                drop_last=True, 
                                num_workers=num_workers)
    else:
        # test dataloader
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                drop_last=True, 
                                num_workers=num_workers)
    
    return dataloader


def get_core(model_name, 
             n_classes,
             device,
             parallelize,
             optimizer_name, 
             lr,
             momentum,
             weight_decay,
             loss_fn_name,
             ignore_index):
    
    if model_name == 'DeepLabV2':
        model = get_deeplab_v2(num_classes=n_classes, pretrain=True, pretrain_model_path=DEEPLABV2_PATH).to(device)
        if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(device)
    elif model_name == 'BiSeNet':
        model = BiSeNet(num_classes=n_classes, context_path="resnet18").to(device)
        if parallelize and device == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(device)
    else:
        print('Model accepted: [DeepLabV2, BiSeNet]')
            
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=lr, 
                                    momentum=momentum, 
                                    weight_decay=weight_decay)
    else:
        print('Optimizer accepted: [Adam, SGD]')
        
    if loss_fn_name == 'CrossEntropyLoss':
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        print('Loss function accepted: [CrossEntropyLoss]')
        
    return model, optimizer, loss_fn

def get_loaders(train_dataset_name, 
                val_dataset_name, 
                augumented,
                batch_size,
                n_workers):

    image_transform = transforms.Compose([ # transforms OR albumentation????
        transforms.Resize((512,1024)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    label_transform = transforms.Compose([
        transforms.Resize((512,1024)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if train_dataset_name == 'CityScapes':
        train_dataset = CityScapes(root_dir='./data/Cityscapes', 
                                   split='train', 
                                   image_transform=image_transform, 
                                   label_transform=label_transform)
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=n_workers)
    elif train_dataset_name == 'GTA5':
        train_dataset = GTA5(root_dir='./data/GTA5', 
                             image_transform=image_transform, 
                             label_transform=label_transform)
        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=n_workers)
    else:
        print('Train datasets accepted: [CityScapes, GTA5]')
        
    if val_dataset_name == 'CityScapes':
        val_dataset = CityScapes(root_dir='./data/Cityscapes', 
                                 split='val', 
                                 image_transform=image_transform, 
                                 label_transform=label_transform)
        
        val_loader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=n_workers)
        
        data_height = CITYSCAPES['height']
        data_width = CITYSCAPES['width']
    else:
        print('Val datasets accepted: [CityScapes]')
    
    return train_loader, val_loader, data_height, data_width
    
    

def pipeline (model_name: str, 
              train_dataset_name: str, 
              val_dataset_name: str,
              n_classes:int,
              epochs: int,
              augumented: bool,
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
              power:str
              ):
    
    # get model
    model, optimizer, loss_fn = get_core(model_name, 
                                         n_classes,
                                         device,
                                         parallelize,
                                         optimizer_name, 
                                         lr,
                                         momentum,
                                         weight_decay,
                                         loss_fn_name,
                                         ignore_index)
    # get loader
    train_loader, val_loader, data_height, data_width = get_loaders(train_dataset_name, 
                                                                    val_dataset_name, 
                                                                    augumented,
                                                                    batch_size,
                                                                    n_workers)
    # train
    model_results = train(model=model, 
                          model_name=model_name,
                          optimizer=optimizer, 
                          loss_fn=loss_fn, 
                          train_loader=train_loader, 
                          val_loader=val_loader, 
                          epochs=epochs, 
                          device=device, 
                          checkpoint_root=checkpoint_root,
                          project_step=project_step,
                          verbose=verbose,
                          n_classes=n_classes,
                          power=power)
    
    #               TODO
    # evaluationTODO
    model_info = compute_flops()
    model_performance = get_latency_and_fps()
    # visualizationTODO
    plot_loss(model_results, model_name, project_step, train_dataset_name, val_dataset_name)
    plot_mIoU(model_results, model_name, project_step, train_dataset_name, val_dataset_name)
    plot_IoU(model_results, model_name, project_step, train_dataset_name, val_dataset_name)
    # saveTODO
    save_results(model, model_results, f"{model_name}_metrics_{project_step}", height=data_height, width=data_width, iterations=100, device=device)
    torch.save(model.state_dict(), f"./checkpoints/{model_name}_{project_step}.pth")
    #               END OF THE ENTIRE PIPELINE
