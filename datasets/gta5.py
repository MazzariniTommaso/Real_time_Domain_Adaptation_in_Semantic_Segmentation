from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from typing import Optional, Tuple
from albumentations import Compose
from utils import get_color_to_id


class GTA5(Dataset):
    
    """
    _summary_    
    """
    
    def __init__(self, 
                 root_dir:str,
                 compute_mask:bool=False,
                 transform: Optional[Compose] = None):
        super(GTA5, self).__init__()
        
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.compute_mask = compute_mask
        self.transform = transform
        if self.compute_mask:
            self.color_to_id = get_color_to_id()
        
        # Load the data
        self.data = []
        image_dir = os.path.join(self.root_dir, 'images')
        
        if self.compute_mask:
            label_dir = os.path.join(self.root_dir, 'labels')
        else:
            label_dir = os.path.join(self.root_dir, 'masks')
            
        for filename in os.listdir(image_dir):
            image = os.path.join(image_dir, filename)
            label = os.path.join(label_dir, filename)
            self.data.append((image, label))
        
    def __len__(self)->int:
        
        """_summary_

        Returns:
            int: _description_
        """
        
        return len(self.data)

    def __getitem__(self, idx:int)-> Tuple[torch.Tensor,torch.Tensor]:
        
        """_summary_

        Args:
            idx (int): _description_

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: _description_
        """
        
        image_path, label_path = self.data[idx]

        # Load images and labels or masks
        image = Image.open(image_path).convert('RGB')
        
        if self.compute_mask:
            label = self._rgb_to_label(Image.open(label_path).convert('RGB'))
        else:
            label = Image.open(label_path).convert('L')
            
        image, label = np.array(image), np.array(label)
        
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image, label = transformed['image'], transformed['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).float()/255
        label = torch.from_numpy(label).long()
        return image, label
    
    def _rgb_to_label(self, image:Image.Image)->np.ndarray:
        """_summary_

        Args:
            image (Image.Image): _description_

        Returns:
            np.ndarray: _description_
        """
        
        gray_image = Image.new('L', image.size)
        rgb_pixels = image.load()
        gray_pixels = gray_image.load()
        
        for i in range(image.width):
            for j in range(image.height):
                rgb = rgb_pixels[i,j]
                gray_pixels[i,j] = self.color_to_id.get(rgb,255)
                
        return gray_image
