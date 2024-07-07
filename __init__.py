from pipeline import pipeline
from config import *

"""
# for kaggle:
config = {
    'model_name': 'DeepLabV2', # [DeepLabV2, BiSeNet]
    'train_dataset_name': 'CityScapes', # [CityScapes, GTA5]
    'val_dataset_name': 'CityScapes', # [CityScapes]
    'n_classes': 19,
    'epochs': 50,
    'augmented': False,
    'augmentedType': 'transform1', # [transform1,transform2,transform3,transform4]
    'optimizer_name': 'SGD', # [SGD, Adam]
    'lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'loss_fn_name': 'CrossEntropyLoss', # [CrossEntropyLoss]
    'ignore_index': 255,
    'batch_size': 4, # [2,4,8]
    'n_workers': 4, # [0,2,4]
    'device': 'cuda',
    'parallelize': True,
    'project_step': 'Step2_1', # [Step2_1,Step2_2,Step3_1,Step3_2,Step4]
    'verbose': True,
    'checkpoint_root': 'checkpoints',
    'power': 0.9,
    'evalIterations': 100,
    'ignore_model_measurements': False,           
}

CITYSCAPES_PATH = 'kaggle_directory_for_Cityscapes'
GTA5_PATH = 'kaggle_directory_for_GTA5_without_masks'
GTA5_PATH_WITH_MASK = 'kaggle_directory_for_GTA5_with_masks'
DEEPLABV2_PATH = 'kaggle_directory_for_pretrained_deeplab'
CITYSCAPES = {
    'width': 1024, 
    'height': 512
}
GTA = {
    'width': 1280, 
    'height': 720
}

"""

if __name__ == '__main__':

    """
    # for kaggle:
    pipeline(
        model_name=config['model_name'], 
        train_dataset_name=config['train_dataset_name'], 
        val_dataset_name=config['val_dataset_name'],
        n_classes=config['n_classes'],
        epochs=config['epochs'],
        augmented=config['augmented'],
        augmentedType=config['augmentedType'],
        optimizer_name=config['optimizer_name'],
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        loss_fn_name=config['loss_fn_name'],
        ignore_index=config['ignore_index'],
        batch_size=config['batch_size'],
        n_workers=config['n_workers'],
        device=config['device'],
        parallelize=config['parallelize'],
        project_step=config['project_step'],
        verbose=config['verbose'],
        checkpoint_root=config['checkpoint_root'],
        power=config['power'],
        evalIterations=config['evalIterations'],
        adversarial=config['adversarial']
    )
    """
    
    pipeline(
        model_name=MODEL_NAME, 
        train_dataset_name=TRAIN_DATASET_NAME, 
        val_dataset_name=VAL_DATASET_NAME,
        n_classes=N_CLASSES,
        epochs=EPOCHS,
        augmented=AUGMENTED,
        augmentedType=AUGMENTED_TYPE,
        optimizer_name=OPTIMIZER_NAME,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        loss_fn_name=LOSS_FN_NAME,
        ignore_index=IGNORE_INDEX,
        batch_size=BATCH_SIZE,
        n_workers=N_WORKERS,
        device=DEVICE,
        parallelize=PARALLELIZE,
        project_step=PROJECT_STEP,
        verbose=VERBOSE,
        checkpoint_root=CHECKPOINT_ROOT,
        power=POWER,
        evalIterations=EVAL_ITERATIONS,
        ignore_model_measurements=IGNORE_MODEL_MEASUREMENTS
    ) 
