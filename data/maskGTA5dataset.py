import os
from PIL import Image
from utils import get_color_to_id

def convert_image_to_grayscale(image_path: str, color_to_id: dict, output_path: str):
    """
    Convert a color segmented image to a grayscale segmented image.

    Args:
        image_path (str): Path to the input color segmented image.
        color_to_id (dict): Dictionary mapping RGB color tuples to class IDs.
        output_path (str): Path to save the output grayscale segmented image.
    """
    color_to_id = get_color_to_id()
    image = Image.open(image_path).convert('RGB')
    gray_image = Image.new('L', image.size)
    rgb_pixels = image.load()
    gray_pixels = gray_image.load()

    for i in range(image.width):
        for j in range(image.height):
            rgb = rgb_pixels[i, j]
            gray_pixels[i, j] = color_to_id.get(rgb, 255)  # 255 is the default value for unknown colors

    gray_image.save(output_path)

def process_images(input_folder: str, output_folder: str):
    """
    Process all images in the input folder, converting them to grayscale and saving them in the output folder.

    Args:
        input_folder (str): Path to the folder containing input color segmented images.
        output_folder (str): Path to the folder to save output grayscale segmented images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    color_to_id = get_color_to_id()
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            convert_image_to_grayscale(input_path, color_to_id, output_path)
            print(f"Converted {filename} and saved to {output_path}")

# Directory paths
input_folder = 'data/GTA5_with_mask/labels'
output_folder = 'data/GTA5_with_mask/masks'

# Process images
process_images(input_folder, output_folder)
