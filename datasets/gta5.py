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
    A dataset class for loading and processing the GTA5 dataset.
    """
    
    def __init__(self, 
                 root_dir:str,
                 compute_mask:bool=False,
                 transform: Optional[Compose] = None):
        """
        Initializes the GTA5 dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            compute_mask (bool, optional): Whether to compute the mask from RGB labels. Defaults to False.
            transform (Optional[Compose], optional): Transformations to be applied on images and labels. Defaults to None.
        """
        super(GTA5, self).__init__()
        
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
            
    def _rgb_to_label(self, image:Image.Image)->np.ndarray:
        """
        Converts an RGB image to a label image using the color to ID mapping.

        Args:
            image (Image.Image): The input RGB image.

        Returns:
            np.ndarray: The label image.
        """
        gray_image = Image.new('L', image.size)
        rgb_pixels = image.load()
        gray_pixels = gray_image.load()
        
        for i in range(image.width):
            for j in range(image.height):
                rgb = rgb_pixels[i,j]
                gray_pixels[i,j] = self.color_to_id.get(rgb,255)
                
        return gray_image
        
    def __len__(self)->int: 
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx:int)-> Tuple[torch.Tensor,torch.Tensor]:
        """
        Generates one sample of data.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image and the corresponding label or mask.
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
