from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from typing import Optional, Tuple
from albumentations import Compose

class CityScapes(Dataset):
    
    """
    _summary_
    """
    def __init__(self, 
                 root_dir:str, 
                 split:str = 'train', 
                 image_transform: Optional[Compose] = None, 
                 label_transform: Optional[Compose] = None):
        super(CityScapes, self).__init__()
        
        """
        
        _summary
        
        Args:
            root_dir (string): Directory with all the images and annotations.
            split (string): 'train' or 'val'.
            image_transform (callable, optional): Optional transform to be applied on a sample image.
            label_transform (callable, optional): Optional transform to be applied on a sample label.
        """
        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.data= self._load_data(split)
    
    def _load_data(self, split:str)->list:
        
        """
        
        _summary
        
        Args:
            split (str): _description_

        Returns:
            list: _description_
        """
        
        data = []
        path = os.path.join(self.root_dir, 'images', split)
        for city in os.listdir(path):
            images = os.path.join(path, city)
            for image in os.listdir(images):
                label = image.replace('images', 'gtFine').replace('_leftImg8bit', '_gtFine_labelTrainIds')
                data.append((image, label))
        return data

    def __len__(self)->int:
        
        """
        
        _summary
        
        Returns:
            int: _description_
        """
        
        return len(self.image_files)

    def __getitem__(self, idx:int)-> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        
        _summary
        
        Args:
            idx (int): _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor]: _description_
        """
        
        image_path, label_path = self.data[idx]

        # Load image and label
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        image, label = np.array(image), np.array(label)

        if self.image_transform:
            image = self.image_transform(image) # Only resize
        if self.label_transform:
            label = self.label_transform(label) # Only resize

        image = torch.from_numpy(image).permute(2, 0, 1).float()/255
        label = torch.from_numpy(label).long()
        
        return image, label