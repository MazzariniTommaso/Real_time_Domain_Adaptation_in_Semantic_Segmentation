import numpy as np
from PIL import Image
import PIL
from config import GTA
import albumentations as A

# DA COMMENTARE

def get_color_to_id() -> dict:
    """
    Returns a dictionary mapping RGB color tuples to their corresponding class IDs.

    Returns:
        dict: A dictionary where keys are RGB color tuples and values are class IDs.
    """
    id_to_color = get_id_to_color()
    color_to_id = {color: id for id, color in id_to_color.items()}
    return color_to_id

def get_id_to_color() -> dict:
    """
    Returns a dictionary mapping class IDs to their corresponding colors.

    Returns:
        dict: A dictionary where keys are class IDs and values are RGB color tuples.
    """
    return {
        0: (128, 64, 128),    # road
        1: (244, 35, 232),    # sidewalk
        2: (70, 70, 70),      # building
        3: (102, 102, 156),   # wall
        4: (190, 153, 153),   # fence
        5: (153, 153, 153),   # pole
        6: (250, 170, 30),    # light
        7: (220, 220, 0),     # sign
        8: (107, 142, 35),    # vegetation
        9: (152, 251, 152),   # terrain
        10: (70, 130, 180),   # sky
        11: (220, 20, 60),    # person
        12: (255, 0, 0),      # rider
        13: (0, 0, 142),      # car
        14: (0, 0, 70),       # truck
        15: (0, 60, 100),     # bus
        16: (0, 80, 100),     # train
        17: (0, 0, 230),      # motorcycle
        18: (119, 11, 32),    # bicycle
    }

def get_id_to_label() -> dict:
    """
    Returns a dictionary mapping class IDs to their corresponding labels.

    Returns:
        dict: A dictionary where keys are class IDs and values are labels.
    """
    return {
        0: 'road',
        1: 'sidewalk',
        2: 'building',
        3: 'wall',
        4: 'fence',
        5: 'pole',
        6: 'light',
        7: 'sign',
        8: 'vegetation',
        9: 'terrain',
        10: 'sky',
        11: 'person',
        12: 'rider',
        13: 'car',
        14: 'truck',
        15: 'bus',
        16: 'train',
        17: 'motorcycle',
        18: 'bicycle',
        255: 'unlabeled'
    }

def label_to_rgb(label:np.ndarray, height:int, width:int)->PIL.Image:
    """
    Transforms a label matrix into a corresponding RGB image utilizing a predefined color map.

    This function maps each label identifier in a two-dimensional array to a specific color, thereby generating an RGB image. This is particularly useful for visualizing segmentation results where each label corresponds to a different segment class.

    Parameters:
        label (np.ndarray): A two-dimensional array where each element represents a label identifier.
        height (int): The desired height of the resulting RGB image.
        width (int): The desired width of the resulting RGB image.

    Returns:
        PIL.Image: An image object representing the RGB image constructed from the label matrix.
    """
    id_to_color = get_id_to_color()
    
    height, width = label.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            class_id = label[i, j]
            rgb_image[i, j] = id_to_color.get(class_id, (255, 255, 255))  # Default to white if not found
    pil_image = Image.fromarray(rgb_image, 'RGB')
    return pil_image

def get_augmented_data(augumentedType:str)-> A.Compose:
    """_summary_

    Args:
        augumentedType (str): _description_

    Returns:
        A.Compose: _description_
    """
    
    augmentations = {
        'transform1': A.Compose([
            A.Resize((GTA['height'],GTA['width'])),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5)
        ]),
        'transform2': A.Compose([
            A.Resize((GTA['height'],GTA['width'])),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ColorJitter(p=0.5),
            A.RandomResizedCrop(height=GTA['height'], 
                                width=GTA['width'], 
                                scale=(0.5, 1.0), 
                                ratio=(0.75, 1.333), 
                                p=0.5)
        ]),
        'transform3': A.Compose([
            A.Resize((GTA['height'],GTA['width'])),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ]),
        'transform4': A.Compose([
            A.Resize((GTA['height'],GTA['width'])),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(height=GTA['height'], 
                                width=GTA['width'], 
                                scale=(0.5, 1.0), 
                                ratio=(0.75, 1.333), 
                                p=0.5),
            A.RandomRotate90(p=0.5)
        ]),
    }
    
    if augumentedType in ['transform1','transform2','transform3','transform4']:
        return augmentations[augumentedType]
    else:
        print('Transformarion accepted: [transform1, transform2, transform3, transform4]')
        return A.Compose([
            A.Resize((GTA['height'],GTA['width'])),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])