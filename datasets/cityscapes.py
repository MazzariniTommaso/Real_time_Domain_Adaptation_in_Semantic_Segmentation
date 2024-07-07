from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from typing import Optional, Tuple
from albumentations import Compose


class CityScapes(Dataset):
    
    """
    A dataset class for loading and processing the CityScapes dataset.
    """
    def __init__(self, 
                 root_dir:str, 
                 split:str = 'train', 
                 transform: Optional[Compose] = None):
        """
        Initializes the CityScapes dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            split (str, optional): Dataset split to use ('train', 'val', 'test'). Defaults to 'train'.
            transform (Optional[Compose], optional): Transformations to be applied on images and labels.. Defaults to None.
        """
        super(CityScapes, self).__init__()

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load the data
        self.data = []
        path = os.path.join(self.root_dir, 'images', split)
        for city in os.listdir(path):
            images = os.path.join(path, city)
            for image in os.listdir(images):
                image = os.path.join(images, image)
                label = image.replace('images', 'gtFine').replace('_leftImg8bit','_gtFine_labelTrainIds')
                self.data.append((image, label))

    def __len__(self)->int:   
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """ 
        return len(self.data)

    def __getitem__(self, idx:int)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates one sample of data.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image and the corresponding label.
        """
        image_path, label_path = self.data[idx]

        # Load image and label
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        image, label = np.array(image), np.array(label)
        
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image, label = transformed['image'], transformed['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).float()/255
        label = torch.from_numpy(label).long()
        
        return image, label