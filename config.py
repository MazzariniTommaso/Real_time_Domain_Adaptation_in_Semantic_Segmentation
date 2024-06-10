# Train conf
EPOCHS = 50
N_CLASSES = 19
LR = 5e-4

# Dataloader conf
BATCH_SIZE = [2,4,8]
NUM_WORKERS = [0,2,4,8]

# Datasets size
CITYSCAPES = {
    'width': 1024, 
    'height': 512
}

GTA5 = {
    'width': 1280, 
    'height': 720
}

# Paths
CITYSCAPES_PATH = 'data/Cityscapes/Cityspaces'
GTA_PATH = 'data/GTA5'
DEEPLABV2_PATH = 'models/deeplab_resnet_pretrained_imagenet.pth'