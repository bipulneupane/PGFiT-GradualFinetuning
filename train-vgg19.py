import os, cv2
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize
import albumentations as albu

import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from utils import *

def train_unet_encoders(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, prefix):
    """
    Trains a U-Net model with a specified encoder and saves the model with the best validation score.

    Args:
        model_name (str): Name of the U-Net model architecture.
        bb (str): Backbone encoder for the U-Net model.
        data_type (str): Dataset type (e.g., the folder name where training, validation, and test data are stored).
        CLASSES (list): List of target class names for segmentation.
        BATCH_SIZE (int): Batch size for training and validation.
        EPOCHS (int): Number of epochs to train the model.
        LR (float): Learning rate for the optimizer.
        optimiser (str): Name of the optimizer to use (e.g., 'adam', 'sgd').
        loss_func (str): Name of the loss function to use (e.g., 'bce', 'dice').
        prefix (str): Prefix for naming the saved model checkpoint.

    Returns:
        None: The function saves the best model checkpoint to the specified path.

    Process:
        1. Initializes the encoder, preprocessing function, and dataset paths.
        2. Sets up training and validation datasets and data loaders.
        3. Configures the U-Net model, loss function, optimizer, and scheduler.
        4. Trains the model for the specified number of epochs.
        5. Validates the model after each epoch and saves the best-performing model based on IoU score.
        6. Prints training progress, metrics, and details about the model and dataset.

    Model Save Path:
        The trained model is saved in the `./trained_models/SDA-strategies/` directory with a file name
        generated using the following pattern:
        `{prefix}{model_name}-{bb}-{data_type}-{EPOCHS}ep-{optimiser}-{loss_func}.pth`
    """
    ENCODER = bb
    ENCODER_WEIGHTS = None
    CLASSES = ['building']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save = model_name + '-' + ENCODER
    checkpoint_path = './trained_models/SDA-strategies/'+str(prefix)+model_save+'-'+data_type+'-'+str(EPOCHS)+'ep'+'-'+str(optimiser)+'-'+str(loss_func)+'.pth'

    ####### DATASET GENERATOR
    DATA_DIR = './data/'+data_type+'/'  
    x_train_dir, x_valid_dir, x_test_dir = os.path.join(DATA_DIR, 'train', 'image'), os.path.join(DATA_DIR, 'val', 'image'), os.path.join(DATA_DIR, 'test', 'image')
    y_train_dir, y_valid_dir, y_test_dir = os.path.join(DATA_DIR, 'train', 'label'), os.path.join(DATA_DIR, 'val', 'label'), os.path.join(DATA_DIR, 'test', 'label')
    # Dataset for train and val images
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) #num_workers=12
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False) #num_workers=4

    ####### COMPILE MODEL
    # create segmentation model with pretrained encoder
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'
    model = get_model_from_smp(model_name, ENCODER, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    model_params=model.parameters()
    params = sum(p.numel() for p in model.parameters())
    
    print()
    print("********************************************************************************")
    print("********************************************************************************")
    print("Training model ", model_name)
    print("Encoder: ", ENCODER)
    print("Network params", params)
    print("Dataset: ", data_type)
    print("Task: Trained alone/pre-train (No SDA)")
    print("Classes: ", CLASSES)
    print("Loss: ", loss_func)
    print("Optimiser: ", optimiser)

    metrics = [smp.utils.metrics.Precision(threshold=0.5), smp.utils.metrics.Recall(threshold=0.5), smp.utils.metrics.IoU(threshold=0.5), smp.utils.metrics.Fscore(threshold=0.5)]
    loss = get_loss(loss_func)
    optim = get_optim(optimiser, model_params, LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max')

    train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True,)
    valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True,)

    max_score = 0

    for i in range(0, EPOCHS):

        print('\nEpoch: {}/{}'.format(i+1,EPOCHS))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1], list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr']))
        
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path) 
            print('Model saved!')
            scheduler.step(max_score)
    
    print("********************************************************************************")
    print("********************************************************************************")
    print()
    print()




def bonai_multilabel_training(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, ckpt_prefix):
    """
    Trains a multi-label segmentation model for footprint and roof detection using the specified configuration.

    Args:
        model_name (str): Name of the model architecture (e.g., U-Net).
        bb (str): Backbone encoder for the model.
        data_type (str): Dataset type (e.g., folder containing training, validation, and test data).
        CLASSES (list): List of target class names for multi-label segmentation.
        BATCH_SIZE (int): Batch size for training and validation.
        EPOCHS (int): Number of training epochs.
        LR (float): Learning rate for the optimizer.
        optimiser (str): Name of the optimizer to use (e.g., 'adam', 'sgd').
        loss_func (str): Name of the loss function to use (e.g., 'jaccard_loss', 'dice_loss').
        ckpt_prefix (str): Prefix for naming the model checkpoint.

    Returns:
        None: The function saves the best model checkpoint to the specified path.

    Process:
        1. Sets up the backbone encoder and initializes the dataset directories.
        2. Configures training and validation datasets using the `BONAIDatasetFootprintRoof` class.
        3. Creates data loaders for efficient training and validation.
        4. Configures the segmentation model with multi-label support using the specified activation function.
        5. Calculates the total number of model parameters and sets up loss function, optimizer, and scheduler.
        6. Trains the model over the specified number of epochs.
        7. Logs training and validation metrics, including precision, recall, IoU, F1-score, and loss.
        8. Saves the best model checkpoint based on IoU score to the `./trained_models/SDA-strategies/` directory.
        9. Adjusts the learning rate dynamically using a scheduler.

    Model Save Path:
        The trained model is saved as:
        `{ckpt_prefix}{model_name}-{bb}-{data_type}-{EPOCHS}ep-{optimiser}-{loss_func}.pth`
    """
    ENCODER = bb
    ENCODER_WEIGHTS = None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save = model_name + '-' + ENCODER
    ckpt_name = './trained_models/SDA-strategies/'+str(ckpt_prefix)+model_save+'-'+data_type+'-'+str(EPOCHS)+'ep'+'-'+str(optimiser)+'-'+str(loss_func)+'.pth'

    transform = Compose([
        ToTensor(),  # Only keep ToTensor in transforms
    ])

    ####### DATASET GENERATOR
    DATA_DIR = './data/'+data_type+'/'   
    x_train_dir, x_valid_dir, x_test_dir = os.path.join(DATA_DIR, 'train', 'image'), os.path.join(DATA_DIR, 'val', 'image'), os.path.join(DATA_DIR, 'test', 'image')
    y_train_dir, y_valid_dir, y_test_dir = os.path.join(DATA_DIR, 'train', 'label'), os.path.join(DATA_DIR, 'val', 'label'), os.path.join(DATA_DIR, 'test', 'label')
    
    train_dataset = BONAIDatasetFootprintRoof(img_dir=x_train_dir, ann_dir=y_train_dir, transform=transform)
    valid_dataset = BONAIDatasetFootprintRoof(img_dir=x_valid_dir, ann_dir=y_valid_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) #num_workers=12
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False) #num_workers=4

    ####### COMPILE MODEL
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    #ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax2d'
    ACTIVATION = 'sigmoid'
    model = get_model_from_smp(model_name, ENCODER, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    model_params=model.parameters()
    params = sum(p.numel() for p in model.parameters())
    
    print()
    print("********************************************************************************")
    print("********************************************************************************")
    print("Training model ", model_name)
    print("Encoder: ", ENCODER)
    print("Network params: ", params)
    print("Dataset: ", data_type)
    print("Task: MultiLabel Pretraining (Roof/Footprint)")
    print("Classes: ", CLASSES)
    print("Loss: ", loss_func)
    print("Optimiser: ", optimiser)

    metrics = [
        smp.utils.metrics.Precision(threshold=0.5), 
        smp.utils.metrics.Recall(threshold=0.5), 
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]
    
    loss = get_multilabel_loss(loss_func)
    
    if loss_func == 'jaccard_loss':
        loss.__name__ = 'Jaccard_loss'
    elif loss_func == 'dice_loss':
        loss.__name__ = 'Dice_loss'
    else:
        print('Loss name is wrong.')
    

    optim = get_optim(optimiser, model_params, LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max')

    train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True,)
    valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True,)

    max_score = 0
    for i in range(0, EPOCHS):

        print('\nEpoch: {}/{}'.format(i+1,EPOCHS))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1], list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr']))
        
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save({'model_state_dict': model.state_dict()}, ckpt_name) 
            print('Model saved!')
            scheduler.step(max_score)
    
    print("********************************************************************************")
    print("********************************************************************************")
    print()
    print()


##### FINE-TUNING - BY BLOCK NAME

def freeze_vgg19_layers_from_block(model, block_name):
    """
    Freezes or unfreezes layers in a VGG19-based model based on the specified block name.

    Args:
        model (torch.nn.Module): The model instance containing the VGG19 encoder.
        block_name (str): Name of the block up to which layers should be unfrozen.
            Options include:
            - "decoder"
            - "bottleneck"
            - "Block 4"
            - "Block 3"
            - "Block 2"
            - "Block 1"
            - "Block 0"

    Returns:
        None: Updates the `requires_grad` attribute of model parameters in-place.

    Process:
        1. **Freeze All Layers**: Initially sets `requires_grad=False` for all model parameters.
        2. **Unfreeze Decoder Layers**: Always unfreezes decoder layers and segmentation head if the block name is "decoder" or higher.
        3. **Unfreeze Bottleneck Layers**: If the block name is "bottleneck" or higher, also unfreezes bottleneck layers.
        4. **Unfreeze Encoder Blocks**: Gradually unfreezes encoder blocks based on the specified `block_name`. Each block includes specific layers, as defined in the `block_layers` dictionary.
        5. Prints the layers or blocks unfrozen after each step for transparency.

    Block Structure:
        - **Block 0**: Initial convolutional layers.
        - **Block 1-4**: Deeper convolutional layers in the VGG19 encoder.
        - **Bottleneck**: Bottleneck layer connecting the encoder and decoder.
        - **Decoder**: All layers in the decoder and segmentation head.

    Example:
        >>> model = get_model_from_smp("Unet", "vgg19", None, len(CLASSES), "sigmoid")
        >>> freeze_vgg19_layers_from_block(model, "Block 2")
        Unfrozen up to decoder
        Unfrozen up to bottleneck
        Unfrozen up to Block 2
    """
    # Dictionary to map block names to corresponding layer indices in VGG19 encoder
    block_layers = {
        'bottleneck': ['decoder.center.0', 'decoder.center.1'],
        'Block 4': ['encoder.features.28', 'encoder.features.30', 'encoder.features.32', 'encoder.features.34'],
        'Block 3': ['encoder.features.19', 'encoder.features.21', 'encoder.features.23', 'encoder.features.25'],
        'Block 2': ['encoder.features.10', 'encoder.features.12', 'encoder.features.14', 'encoder.features.16'],
        'Block 1': ['encoder.features.5', 'encoder.features.7'],
        'Block 0': ['encoder.features.0', 'encoder.features.2']
    }

    # First, freeze all layers in the model
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze the decoder layers if we are working on 'decoder' or higher
    if block_name in ['decoder', 'bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0']:
        for name, param in model.named_parameters():
            if name.startswith("decoder.blocks") or name.startswith("segmentation_head"):
                param.requires_grad = True
        print("Unfrozen up to decoder")

    # Unfreeze bottleneck layers if we are working on 'bottleneck' or higher
    if block_name in ['bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0']:
        for name, param in model.named_parameters():
            if any(bn_layer in name for bn_layer in block_layers['bottleneck']):
                param.requires_grad = True
        print("Unfrozen up to bottleneck")

    # Unfreeze encoder blocks progressively based on the specified block_name
    blocks_to_unfreeze = {
        'Block 4': ['Block 4'],
        'Block 3': ['Block 3', 'Block 4'],
        'Block 2': ['Block 2', 'Block 3', 'Block 4'],
        'Block 1': ['Block 1', 'Block 2', 'Block 3', 'Block 4'],
        'Block 0': ['Block 0', 'Block 1', 'Block 2', 'Block 3', 'Block 4']
    }

    if block_name in blocks_to_unfreeze:
        for block in blocks_to_unfreeze[block_name]:
            for layer_name in block_layers[block]:
                for name, param in model.named_parameters():
                    if name.startswith(layer_name):
                        param.requires_grad = True
        print(f"Unfrozen up to {block_name}")




# Fine-tuning function
def fine_tune_unet_vgg19(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, checkpoint_path, block_name):
    """
    Fine-tunes a U-Net model with a VGG19 encoder up to a specified block, using a given dataset and training configuration.

    Args:
        model_name (str): Name of the U-Net model architecture.
        bb (str): Backbone encoder (e.g., 'vgg19').
        data_type (str): Dataset type (e.g., folder name for training, validation, and test data).
        CLASSES (list): List of target class names for segmentation.
        BATCH_SIZE (int): Batch size for training and validation.
        EPOCHS (int): Number of training epochs.
        LR (float): Learning rate for the optimizer.
        optimiser (str): Name of the optimizer to use (e.g., 'adam', 'sgd').
        loss_func (str): Loss function to use (e.g., 'bce', 'dice').
        checkpoint_path (str): Path to the model checkpoint for loading pre-trained weights.
        block_name (str): Name of the encoder block up to which layers should be unfrozen. Options include:
            - "decoder"
            - "bottleneck"
            - "Block 4"
            - "Block 3"
            - "Block 2"
            - "Block 1"
            - "Block 0"

    Returns:
        None: Trains the model and saves the best-performing model checkpoint.

    Process:
        1. **Dataset Preparation**: Initializes dataset directories and creates data loaders for training and validation.
        2. **Model Compilation**:
            - Builds a U-Net model with the specified encoder.
            - Loads pre-trained weights from the given checkpoint (if available).
        3. **Layer Freezing**: Freezes model layers based on the specified `block_name` using the `freeze_vgg19_layers_from_block` function.
        4. **Training Configuration**:
            - Configures loss function, metrics (Precision, Recall, IoU, F1), and optimizer.
            - Implements early stopping with a patience of 11 epochs.
        5. **Training Loop**:
            - Trains the model for the specified number of epochs.
            - Evaluates the model on the validation dataset at each epoch.
            - Saves the best model checkpoint based on the highest IoU score.
            - Stops training if early stopping conditions are met.
        6. **Logging**:
            - Prints model configuration details, training progress, and validation metrics.

    Model Save Path:
        The trained model is saved at:
        `./trained_models/SDA-strategies/NormalSDA-upto-{block_name}-{model_name}-{bb}-{data_type}-{EPOCHS}ep-{optimiser}-{loss_func}-LR5.pth`

    Example:
        >>> fine_tune_unet_vgg19(
        ...     model_name="Unet",
        ...     bb="vgg19",
        ...     data_type="segmentation",
        ...     CLASSES=["building"],
        ...     BATCH_SIZE=16,
        ...     EPOCHS=50,
        ...     LR=0.001,
        ...     optimiser="adam",
        ...     loss_func="bce",
        ...     checkpoint_path="./checkpoint.pth",
        ...     block_name="Block 3"
        ... )
    """
    ENCODER = bb
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Change to 'mps' if using Apple M1/M2
    model_save = model_name + '-' + ENCODER
    checkpoint_save_path = './trained_models/SDA-strategies/NormalSDA-upto-'+ str(block_name.replace(' ', '')) + '-' + model_save + '-' + data_type + '-' + str(EPOCHS) + 'ep-' + str(optimiser) + '-' + str(loss_func) + '-' + 'LR5' + '.pth'

    ####### DATASET GENERATOR
    DATA_DIR = './data/' + data_type + '/'
    x_train_dir = os.path.join(DATA_DIR, 'train', 'image')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'image')
    x_test_dir = os.path.join(DATA_DIR, 'test', 'image')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'label')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'label')
    y_test_dir = os.path.join(DATA_DIR, 'test', 'label')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    ####### COMPILE MODEL
    n_classes = 1 if len(CLASSES) == 1 else len(CLASSES)
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'
    
    # Create the U-Net model
    #model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, classes=n_classes, activation=ACTIVATION)
    model = get_model_from_smp(model_name, ENCODER, None, len(CLASSES), ACTIVATION)

    print()
    print("********************************************************************************")
    print("Task: Normal Fine-tuning upto a specified block name")

    print(f"Loading model weights from checkpoint: {checkpoint_path}")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Training from scratch!")

    # Freeze layers based on the block parameter
    freeze_vgg19_layers_from_block(model, block_name)

    ####### PRINT MODEL PARAMETERS
    model_params = model.parameters()
    params = sum(p.numel() for p in model.parameters())
    print(f"Training model: {model_name}")
    print(f"Encoder: {ENCODER}")
    print(f"Unfrozen Upto: {block_name}")
    print(f"Total parameters: {params}")
    print(f"Dataset: {data_type}")
    print(f"Loss: {loss_func}")
    print(f"Optimiser: {optimiser}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    ####### LOSS, METRICS, AND OPTIMIZER
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    loss = get_loss(loss_func)
    optim = get_optim(optimiser, model_params, LR)

    ####### EARLY STOPPING SETUP
    early_stopping_patience = 11  # Stop if no improvement in IoU for 11 epochs
    early_stopping_counter = 0
    max_score = 0

    ####### TRAINING LOOP
    train_epoch = smp.utils.train.TrainEpoch(
        model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=DEVICE, verbose=True
    )

    for i in range(EPOCHS):
        print(f'\nEpoch: {i+1}/{EPOCHS}')
        
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)
        print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1], list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr']))
        
        # Early stopping and model saving
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            early_stopping_counter = 0  # Reset counter if IoU improves
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_save_path)
            print('Model saved!')
        else:
            early_stopping_counter += 1
            print(f"No improvement in IoU. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        # Check early stopping condition
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered. Stopping training.")
            break

    print("Training complete!")
    print("********************************************************************************")
    print("********************************************************************************")
    print()
    print()


##### GRADUAL FINE-TUNING

def print_trainable_params(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}/{total_params} (Trainable/Total)")

def gradual_fine_tune_unet_vgg19(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, checkpoint_path, postfix):
    """
    Gradually fine-tunes a U-Net model with a VGG19 encoder, progressively unfreezing layers up to "Block 0" 
    based on IoU stagnation for 5 epochs.

    Args:
        model_name (str): Name of the U-Net model architecture.
        bb (str): Backbone encoder (e.g., 'vgg19').
        data_type (str): Dataset type (e.g., folder containing training and validation data).
        CLASSES (list): List of target class names for segmentation.
        BATCH_SIZE (int): Batch size for training and validation.
        EPOCHS (int): Total number of epochs for all phases.
        LR (float): Base learning rate for the optimizer.
        optimiser (str): Name of the optimizer to use (e.g., 'adam', 'sgd').
        loss_func (str): Loss function to use (e.g., 'bce', 'dice').
        checkpoint_path (str): Path to the checkpoint for loading pre-trained weights.
        postfix (str): Additional string appended to the checkpoint file name.

    Returns:
        None: Trains the model and saves the best-performing model checkpoint.

    Process:
        1. **Dataset Preparation**: Initializes dataset directories and creates data loaders for training and validation.
        2. **Model Compilation**:
            - Builds a U-Net model with the specified encoder and activation.
            - Loads pre-trained weights from the given checkpoint (if available).
        3. **Gradual Fine-Tuning**:
            - Implements a multi-phase training approach where layers are progressively unfrozen:
              - Phases: ['decoder', 'bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0'].
            - Each phase continues until IoU stagnates for 5 consecutive epochs.
            - Adjusts the optimizer learning rate for each phase and reinitializes phase-specific counters.
        4. **Training Loop**:
            - Trains the model for the specified number of epochs across all phases.
            - Saves the best-performing model checkpoint based on IoU score during each phase.
            - Stops training early if IoU does not improve for 6 consecutive epochs across all phases.
        5. **Logging**:
            - Prints training progress, phase transitions, and validation metrics, including precision, recall, IoU, F1, and loss.

    Model Save Path:
        The trained model is saved at:
        `./trained_models/SDA-strategies/GradualSDA-upto-Block0-{model_name}-{bb}-{data_type}-{EPOCHS}ep-{optimiser}-{loss_func}-{postfix}.pth`

    Example:
        >>> gradual_fine_tune_unet_vgg19(
        ...     model_name="Unet",
        ...     bb="vgg19",
        ...     data_type="segmentation",
        ...     CLASSES=["building"],
        ...     BATCH_SIZE=16,
        ...     EPOCHS=50,
        ...     LR=0.001,
        ...     optimiser="adam",
        ...     loss_func="dice",
        ...     checkpoint_path="./checkpoint.pth",
        ...     postfix="trial1"
        ... )
    """
    ENCODER = bb
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save = model_name + '-' + ENCODER
    checkpoint_save_path = './trained_models/SDA-strategies/GradualSDA-upto-Block0-' + model_save + '-' + data_type + '-' + str(EPOCHS) + 'ep-' + str(optimiser) + '-' + str(loss_func) + '-' + str(postfix) + '.pth'

    ####### DATASET GENERATOR
    DATA_DIR = './data/' + data_type + '/'
    x_train_dir = os.path.join(DATA_DIR, 'train', 'image')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'image')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'label')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'label')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    ####### COMPILE MODEL
    n_classes = 1 if len(CLASSES) == 1 else len(CLASSES)
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'

    # Create the U-Net model
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, classes=n_classes, activation=ACTIVATION)

    print()
    print("********************************************************************************")
    print("Task: Gradual Fine-tuning upto a Block 0 upon IoU stagnation for 5 epochs")

    print(f"Loading model weights from checkpoint: {checkpoint_path}")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Training from scratch!")

    model_params = model.parameters()
    params = sum(p.numel() for p in model.parameters())
    print(f"Training model: {model_name}")
    print(f"Encoder: {ENCODER}")
    print(f"Unfrozen Up to: Block 0")
    print(f"Total parameters: {params}")
    print(f"Dataset: {data_type}")
    print(f"Loss: {loss_func}")
    print(f"Optimiser: {optimiser}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    ####### LOSS, METRICS, AND OPTIMIZER
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    loss = get_loss(loss_func)
    optim = get_optim(optimiser, model.parameters(), LR)

    ####### GRADUAL FINE-TUNING PHASES
    phases = ['decoder', 'bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0']
    patience = 5  # Stagnation patience per phase
    early_stopping_patience = 6
    max_score = 0
    global_epoch = 0  # Track global epoch across phases

    # Phase-wise training loop
    for phase in phases:
        print(f"\nStarting phase: {phase}")
        
        # Set the learning rate constant at LR
        print(f"Setting learning rate to: {LR}")
        
        # Update the optimizer with the new learning rate
        optim = get_optim(optimiser, model.parameters(), LR)

        # Unfreeze the appropriate layers for the current phase
        freeze_vgg19_layers_from_block(model, phase)

        # Print the number of trainable parameters
        print_trainable_params(model)

        # Reset patience counters
        phase_stagnation_counter = 0
        early_stopping_counter = 0

        while True:
            global_epoch += 1  # Continue epoch counting across phases

            print(f'\nEpoch: {global_epoch}/{EPOCHS} (Phase: {phase})')

            # Training and validation loops
            train_epoch = smp.utils.train.TrainEpoch(
                model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True
            )
            valid_epoch = smp.utils.train.ValidEpoch(
                model, loss=loss, metrics=metrics, device=DEVICE, verbose=True
            )

            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(valid_dataloader)

            print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1], list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr']))

            # Save the model only if IoU strictly improves (greater than max_score)
            if valid_logs['iou_score'] > max_score:
                max_score = valid_logs['iou_score']
                phase_stagnation_counter = 0  # Reset stagnation counter
                early_stopping_counter = 0  # Reset early stopping counter
                #torch.save({'model_state_dict': model.state_dict()}, checkpoint_save_path) 
                print(f'Model saved at {checkpoint_save_path}!')
            else:
                # If IoU remains the same or decreases, increment stagnation and early stopping counters
                phase_stagnation_counter += 1
                early_stopping_counter += 1
                print(f"No IoU improvement. Phase stagnation counter: {phase_stagnation_counter}/{patience}")
                print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

            # Stop the phase if IoU stagnates for 5 epochs
            if phase_stagnation_counter >= patience:
                print(f"IoU has stagnated for {patience} consecutive epochs in phase: {phase}. Moving to the next phase.")
                break

            # Early stopping if IoU does not improve for 11 epochs
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered. Stopping training.")
                return  # Exit the entire training process

    print("Training complete!")
    print("********************************************************************************")
    print("********************************************************************************")
    print()
    print()



def bonai_multilabel_gradual_fine_tune_unet_vgg19(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, checkpoint_path, postfix):
    """
    Gradually fine-tunes a multi-label U-Net model with a VGG19 encoder for footprint and roof segmentation,
    progressively unfreezing layers up to "Block 0" based on IoU stagnation for 5 epochs.

    Args:
        model_name (str): Name of the U-Net model architecture.
        bb (str): Backbone encoder (e.g., 'vgg19').
        data_type (str): Dataset type (e.g., folder containing training and validation data).
        CLASSES (list): List of target class names for segmentation (e.g., ['footprint', 'roof']).
        BATCH_SIZE (int): Batch size for training and validation.
        EPOCHS (int): Total number of epochs for all phases.
        LR (float): Base learning rate for the optimizer.
        optimiser (str): Name of the optimizer to use (e.g., 'adam', 'sgd').
        loss_func (str): Loss function to use (e.g., 'jaccard_loss', 'dice_loss').
        checkpoint_path (str): Path to the checkpoint for loading pre-trained weights.
        postfix (str): Additional string appended to the checkpoint file name.

    Returns:
        None: Trains the model and saves the best-performing model checkpoint.

    Process:
        1. **Dataset Preparation**:
            - Initializes dataset directories and creates data loaders for training and validation.
            - Uses `BONAIDatasetFootprintRoof` for multi-label segmentation.
        2. **Model Compilation**:
            - Builds a U-Net model with the specified encoder and activation (`sigmoid` for multi-label tasks).
            - Loads pre-trained weights from the checkpoint if available.
        3. **Gradual Fine-Tuning**:
            - Implements a multi-phase training approach, progressively unfreezing layers:
              - Phases: ['decoder', 'bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0'].
            - Each phase continues until IoU stagnates for 5 consecutive epochs.
            - Adjusts optimizer learning rate for each phase and reinitializes counters for phase-specific stagnation.
        4. **Training Loop**:
            - Trains the model across all phases, saving the best checkpoint based on IoU score.
            - Stops training early if IoU does not improve for 6 consecutive epochs across all phases.
        5. **Logging**:
            - Prints training progress, phase transitions, and validation metrics (Precision, Recall, IoU, F1, and Loss).

    Model Save Path:
        The trained model is saved at:
        `./trained_models/SDA-strategies/GradualSDA-upto-Block0-{model_name}-{bb}-{data_type}-{EPOCHS}ep-{optimiser}-{loss_func}-{postfix}.pth`

    Example:
        >>> bonai_multilabel_gradual_fine_tune_unet_vgg19(
        ...     model_name="Unet",
        ...     bb="vgg19",
        ...     data_type="segmentation",
        ...     CLASSES=["footprint", "roof"],
        ...     BATCH_SIZE=16,
        ...     EPOCHS=50,
        ...     LR=0.001,
        ...     optimiser="adam",
        ...     loss_func="dice_loss",
        ...     checkpoint_path="./checkpoint.pth",
        ...     postfix="trial1"
        ... )
    """
    ENCODER = bb
    ENCODER_WEIGHTS = 'imagenet'  # 'imagenet' pre-trained weights
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save = model_name + '-' + ENCODER
    checkpoint_save_path = './trained_models/SDA-strategies/GradualSDA-upto-Block0-' + model_save + '-' + data_type + '-' + str(EPOCHS) + 'ep-' + str(optimiser) + '-' + str(loss_func) + '-' + str(postfix) + '.pth'

    transform = Compose([
        ToTensor(),  # Only keep ToTensor in transforms
    ])

    ####### DATASET GENERATOR
    DATA_DIR = './data/' + data_type + '/'
    x_train_dir = os.path.join(DATA_DIR, 'train', 'image')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'image')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'label')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'label')

    #preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    train_dataset = BONAIDatasetFootprintRoof(img_dir=x_train_dir, ann_dir=y_train_dir, transform=transform)
    valid_dataset = BONAIDatasetFootprintRoof(img_dir=x_valid_dir, ann_dir=y_valid_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    ####### COMPILE MODEL
    n_classes = 1 if len(CLASSES) == 1 else len(CLASSES)
    ACTIVATION = 'sigmoid'

    # Create the U-Net model
    model = get_model_from_smp(model_name, ENCODER, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)

    print()
    print("********************************************************************************")
    print("Task: Multilabel (footprint, roof) - Gradual Fine-tuning upto a Block 0 upon IoU stagnation for 5 epochs")

    print(f"Loading model weights from checkpoint: {checkpoint_path}")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Training from scratch!")

    model_params = model.parameters()
    params = sum(p.numel() for p in model.parameters())
    print(f"Training model: {model_name}")
    print(f"Encoder: {ENCODER}")
    print(f"Unfrozen Up to: Block 0")
    print(f"Total parameters: {params}")
    print(f"Dataset: {data_type}")
    print(f"Loss: {loss_func}")
    print(f"Optimiser: {optimiser}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    ####### LOSS, METRICS, AND OPTIMIZER
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    loss = get_multilabel_loss(loss_func)
    
    if loss_func == 'jaccard_loss':
        loss.__name__ = 'Jaccard_loss'
    elif loss_func == 'dice_loss':
        loss.__name__ = 'Dice_loss'
    else:
        print('Loss name is wrong.')

    optim = get_optim(optimiser, model.parameters(), LR)

    ####### GRADUAL FINE-TUNING PHASES
    phases = ['decoder', 'bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0']
    patience = 5  # Stagnation patience per phase
    early_stopping_patience = 6
    max_score = 0
    global_epoch = 0  # Track global epoch across phases

    # Phase-wise training loop
    for phase in phases:
        print(f"\nStarting phase: {phase}")
        
        # Set the learning rate constant at LR
        print(f"Setting learning rate to: {LR}")
        
        # Update the optimizer with the new learning rate
        optim = get_optim(optimiser, model.parameters(), LR)

        # Unfreeze the appropriate layers for the current phase
        freeze_vgg19_layers_from_block(model, phase)

        # Print the number of trainable parameters
        print_trainable_params(model)

        # Reset patience counters
        phase_stagnation_counter = 0
        early_stopping_counter = 0

        while True:
            global_epoch += 1  # Continue epoch counting across phases

            print(f'\nEpoch: {global_epoch}/{EPOCHS} (Phase: {phase})')

            # Training and validation loops
            train_epoch = smp.utils.train.TrainEpoch(
                model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True
            )
            valid_epoch = smp.utils.train.ValidEpoch(
                model, loss=loss, metrics=metrics, device=DEVICE, verbose=True
            )

            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(valid_dataloader)

            print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1], list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr']))

            # Save the model only if IoU strictly improves (greater than max_score)
            if valid_logs['iou_score'] > max_score:
                max_score = valid_logs['iou_score']
                phase_stagnation_counter = 0  # Reset stagnation counter
                early_stopping_counter = 0  # Reset early stopping counter
                torch.save({'model_state_dict': model.state_dict()}, checkpoint_save_path)
                print(f'Model saved at {checkpoint_save_path}!')
            else:
                # If IoU remains the same or decreases, increment stagnation and early stopping counters
                phase_stagnation_counter += 1
                early_stopping_counter += 1
                print(f"No IoU improvement. Phase stagnation counter: {phase_stagnation_counter}/{patience}")
                print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

            # Stop the phase if IoU stagnates for 5 epochs
            if phase_stagnation_counter >= patience:
                print(f"IoU has stagnated for {patience} consecutive epochs in phase: {phase}. Moving to the next phase.")
                break

            # Early stopping if IoU does not improve for 11 epochs
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered. Stopping training.")
                return  # Exit the entire training process

    print("Training complete!")
    print("********************************************************************************")
    print("********************************************************************************")
    print()
    print()


########## Gradual Fine tuning + Discriminative LR reduction upon IoU Stagnation for 5 epochs

def gradual_fine_tune_disclr_unet_vgg19(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, checkpoint_path, postfix):
    """
    Our implementation of Discriminative Learning Rate Reduction while Gradual Unfreezing inspired from the following paper.
    @misc{howard2018universallanguagemodelfinetuning,
      title={Universal Language Model Fine-tuning for Text Classification}, 
      author={Jeremy Howard and Sebastian Ruder},
      year={2018},
      eprint={1801.06146},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1801.06146}
      }

    Gradually fine-tunes a U-Net model with a VGG19 encoder using discriminative learning rate (LR) reduction 
    and progressive unfreezing of layers upon IoU stagnation for 5 epochs.

    Args:
        model_name (str): Name of the U-Net model architecture.
        bb (str): Backbone encoder (e.g., 'vgg19').
        data_type (str): Dataset type (e.g., folder containing training and validation data).
        CLASSES (list): List of target class names for segmentation.
        BATCH_SIZE (int): Batch size for training and validation.
        EPOCHS (int): Total number of epochs for all phases.
        LR (float): Base learning rate for the optimizer.
        optimiser (str): Name of the optimizer to use (e.g., 'adam', 'sgd').
        loss_func (str): Loss function to use (e.g., 'bce', 'dice').
        checkpoint_path (str): Path to the checkpoint for loading pre-trained weights.
        postfix (str): Additional string appended to the checkpoint file name.

    Returns:
        None: Trains the model and saves the best-performing model checkpoint.

    Process:
        1. **Dataset Preparation**:
            - Initializes dataset directories and creates data loaders for training and validation.
        2. **Model Compilation**:
            - Builds a U-Net model with the specified encoder and activation.
            - Loads pre-trained weights from the checkpoint if available.
        3. **Gradual Fine-Tuning with Discriminative LR**:
            - Implements a multi-phase training approach with layer-wise unfreezing:
              - Phases: ['decoder', 'bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0'].
            - Adjusts learning rates per phase: progressively reduces LR for deeper layers.
            - Each phase continues until IoU stagnates for 5 consecutive epochs.
        4. **Training Loop**:
            - Trains the model across all phases, saving the best checkpoint based on IoU score.
            - Stops training early if IoU does not improve for 6 consecutive epochs across all phases.
        5. **Logging**:
            - Prints training progress, phase transitions, learning rates, and validation metrics (Precision, Recall, IoU, F1, and Loss).

    Model Save Path:
        The trained model is saved at:
        `./trained_models/SDA-strategies/DiscLR-GradualSDA-upto-Block0-{model_name}-{bb}-{data_type}-{EPOCHS}ep-{optimiser}-{loss_func}-{postfix}.pth`

    Example:
        >>> gradual_fine_tune_disclr_unet_vgg19(
        ...     model_name="Unet",
        ...     bb="vgg19",
        ...     data_type="segmentation",
        ...     CLASSES=["building"],
        ...     BATCH_SIZE=16,
        ...     EPOCHS=50,
        ...     LR=0.001,
        ...     optimiser="adam",
        ...     loss_func="dice",
        ...     checkpoint_path="./checkpoint.pth",
        ...     postfix="trial1"
        ... )
    """
    ENCODER = bb
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save = model_name + '-' + ENCODER
    checkpoint_save_path = './trained_models/SDA-strategies/DiscLR-GradualSDA-upto-Block0-' + model_save + '-' + data_type + '-' + str(EPOCHS) + 'ep-' + str(optimiser) + '-' + str(loss_func) + '-' + str(postfix) + '.pth'

    ####### DATASET GENERATOR
    DATA_DIR = './data/' + data_type + '/'
    x_train_dir = os.path.join(DATA_DIR, 'train', 'image')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'image')
    x_test_dir = os.path.join(DATA_DIR, 'test', 'image')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'label')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'label')
    y_test_dir = os.path.join(DATA_DIR, 'test', 'label')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    ####### COMPILE MODEL
    n_classes = 1 if len(CLASSES) == 1 else len(CLASSES)
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'

    # Create the U-Net model
    #model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, classes=n_classes, activation=ACTIVATION)
    model = get_model_from_smp(model_name, ENCODER, None, len(CLASSES), ACTIVATION)

    print()
    print("********************************************************************************")
    print("Task: Gradual Fine-tuning with Discriminative LR reduction upto a Block 0 upon IoU stagnation for 5 epochs")

    print(f"Loading model weights from checkpoint: {checkpoint_path}")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Training from scratch!")

    model_params = model.parameters()
    params = sum(p.numel() for p in model.parameters())
    print(f"Training model: {model_name}")
    print(f"Encoder: {ENCODER}")
    print(f"Unfrozen Upto: Block 0")
    print(f"Total parameters: {params}")
    print(f"Dataset: {data_type}")
    print(f"Loss: {loss_func}")
    print(f"Optimiser: {optimiser}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    ####### LOSS, METRICS, AND OPTIMIZER
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    loss = get_loss(loss_func)

    ####### GRADUAL FINE-TUNING PHASES
    phases = ['decoder', 'bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0'] 
    patience = 5  # Stagnation patience per phase
    early_stopping_patience = 6
    phase_max_epochs = 5  # Phase-specific max epoch for decoder-only phase
    max_score = 0
    global_epoch = 0  # Track global epoch across phases

    # Phase-wise training loop
    for phase in phases:
        print(f"\nStarting phase: {phase}")
        
        # Adjust the learning rate based on the phase
        if phase == 'decoder':
            current_lr = LR
        elif phase == 'bottleneck':
            current_lr = LR * 0.80
        elif phase == 'Block 4':
            current_lr = LR * 0.60
        elif phase == 'Block 3':
            current_lr = LR * 0.40
        elif phase == 'Block 2':
            current_lr = LR * 0.20
        elif phase == 'Block 1':
            current_lr = LR * 0.10
        elif phase == 'Block 0':
            current_lr = LR * 0.10
        
        print(f"Setting learning rate to: {current_lr}")
        
        # Update the optimizer with the new learning rate
        optim = get_optim(optimiser, model.parameters(), current_lr)

        # Unfreeze the appropriate layers for the current phase
        freeze_vgg19_layers_from_block(model, phase)

        # Print the number of trainable parameters
        print_trainable_params(model)

        # Reset patience counters
        phase_stagnation_counter = 0
        early_stopping_counter = 0

        while True:
            global_epoch += 1  # Continue epoch counting across phases

            print(f'\nEpoch: {global_epoch}/{EPOCHS} (Phase: {phase})')

            # Training and validation loops
            train_epoch = smp.utils.train.TrainEpoch(
                model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True
            )
            valid_epoch = smp.utils.train.ValidEpoch(
                model, loss=loss, metrics=metrics, device=DEVICE, verbose=True
            )

            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(valid_dataloader)

            print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(
                list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1], 
                list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], optim.param_groups[0]['lr']
            ))

            # Save the model only if IoU strictly improves (greater than max_score)
            if valid_logs['iou_score'] > max_score:
                max_score = valid_logs['iou_score']
                phase_stagnation_counter = 0  # Reset stagnation counter
                early_stopping_counter = 0  # Reset early stopping counter
                torch.save({'model_state_dict': model.state_dict()}, checkpoint_save_path)
                print(f'Model saved at {checkpoint_save_path}!')
            else:
                # If IoU remains the same or decreases, increment stagnation and early stopping counters
                phase_stagnation_counter += 1
                early_stopping_counter += 1
                print(f"No IoU improvement. Phase stagnation counter: {phase_stagnation_counter}/{patience}")
                print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

            # Stop the phase if IoU stagnates for 5 epochs
            if phase_stagnation_counter >= patience:
                print(f"IoU has stagnated for {patience} consecutive epochs in phase: {phase}. Moving to the next phase.")
                break

            # Early stopping if IoU does not improve for 11 epochs
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered. Stopping training.")
                return  # Exit the entire training process

    print("Training complete!")
    print("********************************************************************************")
    print("********************************************************************************")
    print()
    print()



def slanted_triangular_lr(current_epoch, max_lr, EPOCHS, cut_frac=0.3, ratio=32):
    """
    Our implementation of Slanted Triangular Learning Rate function from the following paper.
    @misc{howard2018universallanguagemodelfinetuning,
      title={Universal Language Model Fine-tuning for Text Classification}, 
      author={Jeremy Howard and Sebastian Ruder},
      year={2018},
      eprint={1801.06146},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1801.06146}
      }
    
    The function follows a slanted triangular pattern with a rapid increase at the start (for warm-up) 
    and a gradual decay afterward. We pass the epoch and base_lr to compute the learning rate for each epoch dynamically.

    :param current_epoch: Current training step within a phase.
    :param max_lr: Maximum learning rate to reach at the peak.
    :param cut_frac: Fraction of steps to increase the learning rate.
    :param ratio: Ratio by which the minimum learning rate is divided.
    """
    cut = int(cut_frac * EPOCHS)
    if current_epoch < cut:
        p = current_epoch / cut
    else:
        p = 1 - (current_epoch - cut) / (EPOCHS - cut)
    min_lr = max_lr / ratio
    return min_lr + p * (max_lr - min_lr)


def gradual_fine_tune_stlr_unet_vgg19(model_name, bb, data_type, CLASSES, BATCH_SIZE, EPOCHS, LR, optimiser, loss_func, checkpoint_path, postfix):
    """
    Our implementation of Slanted Triangular Learning Rate while Gradual Unfreezing inspired from the following paper.
    @misc{howard2018universallanguagemodelfinetuning,
      title={Universal Language Model Fine-tuning for Text Classification}, 
      author={Jeremy Howard and Sebastian Ruder},
      year={2018},
      eprint={1801.06146},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1801.06146}
      }

    Gradually fine-tunes a U-Net model with a VGG19 encoder using a slanted triangular learning rate (STLR) 
    schedule and progressive unfreezing of layers upon IoU stagnation for 5 epochs.

    Args:
        model_name (str): Name of the U-Net model architecture.
        bb (str): Backbone encoder (e.g., 'vgg19').
        data_type (str): Dataset type (e.g., folder containing training and validation data).
        CLASSES (list): List of target class names for segmentation.
        BATCH_SIZE (int): Batch size for training and validation.
        EPOCHS (int): Total number of epochs for all phases.
        LR (float): Base learning rate for the optimizer.
        optimiser (str): Name of the optimizer to use (e.g., 'adam', 'sgd').
        loss_func (str): Loss function to use (e.g., 'bce', 'dice').
        checkpoint_path (str): Path to the checkpoint for loading pre-trained weights.
        postfix (str): Additional string appended to the checkpoint file name.

    Returns:
        None: Trains the model and saves the best-performing model checkpoint.

    Process:
        1. **Dataset Preparation**:
            - Initializes dataset directories and creates data loaders for training and validation.
        2. **Model Compilation**:
            - Builds a U-Net model with the specified encoder and activation.
            - Loads pre-trained weights from the checkpoint if available.
        3. **Gradual Fine-Tuning with STLR**:
            - Implements a multi-phase training approach, progressively unfreezing layers:
              - Phases: ['decoder', 'bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0'].
            - Adjusts base learning rates per phase (discriminative learning rate strategy).
            - Uses a slanted triangular learning rate schedule within each phase.
            - Each phase continues until IoU stagnates for 5 consecutive epochs.
        4. **Training Loop**:
            - Trains the model across all phases, saving the best checkpoint based on IoU score.
            - Stops training early if IoU does not improve for 6 consecutive epochs across all phases.
        5. **Logging**:
            - Prints training progress, phase transitions, learning rates, and validation metrics (Precision, Recall, IoU, F1, and Loss).

    Model Save Path:
        The trained model is saved at:
        `./trained_models/SDA-strategies/STLR-GradualSDA-upto-Block0-{model_name}-{bb}-{data_type}-{EPOCHS}ep-{optimiser}-{loss_func}-{postfix}.pth`

    Example:
        >>> gradual_fine_tune_stlr_unet_vgg19(
        ...     model_name="Unet",
        ...     bb="vgg19",
        ...     data_type="segmentation",
        ...     CLASSES=["building"],
        ...     BATCH_SIZE=16,
        ...     EPOCHS=50,
        ...     LR=0.001,
        ...     optimiser="adam",
        ...     loss_func="dice",
        ...     checkpoint_path="./checkpoint.pth",
        ...     postfix="trial1"
        ... )
    """
    ENCODER = bb
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save = model_name + '-' + ENCODER
    checkpoint_save_path = './trained_models/SDA-strategies/STLR-GradualSDA-upto-Block0-' + model_save + '-' + data_type + '-' + str(EPOCHS) + 'ep-' + str(optimiser) + '-' + str(loss_func) + '-' + str(postfix) + '.pth'

    ####### DATASET GENERATOR
    DATA_DIR = './data/' + data_type + '/'
    x_train_dir = os.path.join(DATA_DIR, 'train', 'image')
    x_valid_dir = os.path.join(DATA_DIR, 'val', 'image')
    x_test_dir = os.path.join(DATA_DIR, 'test', 'image')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'label')
    y_valid_dir = os.path.join(DATA_DIR, 'val', 'label')
    y_test_dir = os.path.join(DATA_DIR, 'test', 'label')

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    train_dataset = Dataset(x_train_dir, y_train_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn),)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    ####### COMPILE MODEL
    n_classes = 1 if len(CLASSES) == 1 else len(CLASSES)
    ACTIVATION = 'sigmoid' if n_classes == 1 else 'softmax'

    # Create the U-Net model
    #model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, classes=n_classes, activation=ACTIVATION)
    model = get_model_from_smp(model_name, ENCODER, None, len(CLASSES), ACTIVATION)

    print()
    print("********************************************************************************")
    print("Task: Gradual Fine-tuning with STLR with phase shift upon IoU stagnation for 5 epochs")

    print(f"Loading model weights from checkpoint: {checkpoint_path}")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Training from scratch!")

    model_params = model.parameters()
    params = sum(p.numel() for p in model.parameters())
    print(f"Training model: {model_name}")
    print(f"Encoder: {ENCODER}")
    print(f"Unfrozen Upto: Block 0")
    print(f"Total parameters: {params}")
    print(f"Dataset: {data_type}")
    print(f"Loss: {loss_func}")
    print(f"Optimiser: {optimiser}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    ####### LOSS, METRICS, AND OPTIMIZER
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    loss = get_loss(loss_func)

    ####### GRADUAL FINE-TUNING PHASES
    phases = ['decoder', 'bottleneck', 'Block 4', 'Block 3', 'Block 2', 'Block 1', 'Block 0'] 
    patience = 5
    early_stopping_patience = 6
    max_score = 0
    global_epoch = 0

    # Phase-wise training loop
    for phase in phases:
        print(f"\nStarting phase: {phase}")
        
        # Adjust the base learning rate based on the phase (following discriminative learning rate strategy)
        if phase == 'decoder':
            base_lr = LR  # Highest learning rate for decoder phase
        elif phase == 'bottleneck':
            base_lr = LR * 0.80  # Slightly lower learning rate for bottleneck phase
        elif phase == 'Block 4':
            base_lr = LR * 0.60
        elif phase == 'Block 3':
            base_lr = LR * 0.40
        elif phase == 'Block 2':
            base_lr = LR * 0.20
        elif phase == 'Block 1':
            base_lr = LR * 0.10
        elif phase == 'Block 0':
            base_lr = LR * 0.10

        print(f"Base learning rate for phase {phase}: {base_lr}")
        
        # Unfreeze the appropriate layers for the current phase
        freeze_vgg19_layers_from_block(model, phase)

        # Custom optimizer setup for each phase with a slanted triangular schedule
        optim = get_optim(optimiser, model.parameters(), base_lr)

        # Reset counters
        phase_stagnation_counter = 0
        early_stopping_counter = 0

        for epoch in range(EPOCHS):
            global_epoch += 1  # Continue epoch counting across phases
            
            # Slanted Triangular Learning Rate Update with adjusted base_lr for current phase
            current_lr = slanted_triangular_lr(epoch, base_lr, EPOCHS)
            for param_group in optim.param_groups:
                param_group['lr'] = current_lr

            print(f'\nEpoch: {global_epoch}/{EPOCHS} (Phase: {phase}) - LR: {current_lr:.6f}')

            # Training and validation loops
            train_epoch = smp.utils.train.TrainEpoch(
                model, loss=loss, metrics=metrics, optimizer=optim, device=DEVICE, verbose=True
            )
            valid_epoch = smp.utils.train.ValidEpoch(
                model, loss=loss, metrics=metrics, device=DEVICE, verbose=True
            )

            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(valid_dataloader)

            print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}, LR: {}".format(
                list(valid_logs.items())[0][1], list(valid_logs.items())[1][1], list(valid_logs.items())[2][1], 
                list(valid_logs.items())[3][1], list(valid_logs.items())[4][1], current_lr
            ))

            # Save the model if IoU improves
            if valid_logs['iou_score'] > max_score:
                max_score = valid_logs['iou_score']
                phase_stagnation_counter = 0
                early_stopping_counter = 0
                torch.save({'model_state_dict': model.state_dict()}, checkpoint_save_path)
                print(f'Model saved at {checkpoint_save_path}!')
            else:
                # Increment counters on no improvement
                phase_stagnation_counter += 1
                early_stopping_counter += 1
                print(f"No IoU improvement. Phase stagnation counter: {phase_stagnation_counter}/{patience}")
                print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

            # Stop the phase if IoU stagnates for 5 epochs
            if phase_stagnation_counter >= patience:
                print(f"IoU has stagnated for {patience} consecutive epochs in phase: {phase}. Moving to the next phase.")
                break

            # Early stopping if IoU does not improve for 11 epochs
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered. Stopping training.")
                return

    print("Training complete!")
    print("********************************************************************************")



def evaluate_edns(saved_ckpt, edn, bb, data_type, loss_func):
    """
    Evaluates a segmentation model on a validation dataset and computes performance metrics.

    Args:
        saved_ckpt (str): Path to the saved model checkpoint (.pth file).
        edn (str): Name of the encoder-decoder network architecture (e.g., U-Net, FPN).
        bb (str): Backbone encoder name (e.g., 'resnet34', 'vgg19').
        data_type (str): Dataset type specifying the directory structure (e.g., 'building').
        loss_func (str): Name of the loss function used during evaluation (e.g., 'dice_loss', 'jaccard_loss').

    Returns:
        None: Prints the loss and evaluation metrics (Precision, Recall, IoU, and F1 score).

    Process:
        1. **Dataset Preparation**:
            - Loads the validation dataset using the specified preprocessing function.
            - Configures a data loader for batch-wise evaluation.
        2. **Model Loading**:
            - Builds the model with the specified architecture, backbone, and activation function ('sigmoid').
            - Loads pretrained weights from the provided checkpoint file.
        3. **Evaluation Setup**:
            - Configures the loss function and metrics (Precision, Recall, IoU, F1).
            - Uses the `ValidEpoch` utility from `segmentation_models_pytorch` (smp) for evaluation.
        4. **Metrics Calculation**:
            - Evaluates the model on the validation dataset.
            - Computes and displays overall loss and metrics for the dataset.
        5. **Logging**:
            - Prints model configuration details and validation results.

    Example:
        >>> evaluate_edns(
        ...     saved_ckpt="./saved_model.pth",
        ...     edn="Unet",
        ...     bb="vgg19",
        ...     data_type="segmentation",
        ...     loss_func="dice_loss"
        ... )

    Notes:
        - Metrics are computed over the entire validation dataset.
        - The model is evaluated in `eval` mode to ensure no gradient updates.
    """
    ENCODER = bb
    CLASSES=['building']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENCODER_WEIGHTS = None
    ACTIVATION = 'sigmoid'
    print("********************************************************************************")
    print("********************************************************************************")

    ####### DATASET GENERATOR
    DATA_DIR = './data/'+data_type+'/'
    x_train_dir, x_valid_dir, x_test_dir = os.path.join(DATA_DIR, 'train', 'image'), os.path.join(DATA_DIR, 'val', 'image'), os.path.join(DATA_DIR, 'test', 'image')
    y_train_dir, y_valid_dir, y_test_dir = os.path.join(DATA_DIR, 'train', 'label'), os.path.join(DATA_DIR, 'val', 'label'), os.path.join(DATA_DIR, 'test', 'label')
    # Dataset for train and val images
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, classes=CLASSES, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False) #num_workers=4

    ####### Load model, load pretrained Weights, and add into model
    model = get_model_from_smp(edn, bb, ENCODER_WEIGHTS, len(CLASSES), ACTIVATION)
    model_checkpoint = torch.load(saved_ckpt)
    model_pretrained_state_dict = model_checkpoint['model_state_dict']
    model.load_state_dict(model_pretrained_state_dict)
    model.to(DEVICE)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())

    loss = get_loss(loss_func)
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]
    test_epoch = smp.utils.train.ValidEpoch(model=model,loss=loss,metrics=metrics,device=DEVICE,verbose=True,)

    ####### PRINTING SOME DETAILS
    print("Encoder: ", bb)
    print("Checkpoint: ", saved_ckpt)
    print("Validated on: ", data_type)
    print("Net Params: ", params)

    valid_logs = test_epoch.run(valid_dataloader) # on valid folder

    print("Loss: {:.3f}, P: {:.3f}, R: {:.3f}, IoU: {:.3f}, F1: {:.3f}".format(
        list(valid_logs.items())[0][1],
        list(valid_logs.items())[1][1],
        list(valid_logs.items())[2][1],
        list(valid_logs.items())[3][1],
        list(valid_logs.items())[4][1])
         )

    print("********************************************************************************")
    print("********************************************************************************")



def bonai_evaluate_multlabel_edns(saved_ckpt, edn, bb, data_type, classes, loss_func):
    """
    Evaluates a multi-label segmentation model on a validation dataset, calculating class-specific metrics.

    Args:
        saved_ckpt (str): Path to the saved model checkpoint.
        edn (str): Encoder-decoder network architecture name.
        bb (str): Backbone encoder name (e.g., 'vgg19').
        data_type (str): Dataset type (e.g., folder containing training, validation, and test data).
        classes (list): List of class names for multi-label segmentation (e.g., ['footprint', 'roof']).
        loss_func (str): Loss function to use during evaluation (e.g., 'dice_loss', 'jaccard_loss').

    Returns:
        None: Prints class-specific metrics (Precision, Recall, IoU, and F1) for the validation dataset.

    Process:
        1. **Dataset Preparation**:
            - Loads the validation dataset using `BONAIDatasetFootprintRoof` with transformations.
        2. **Model Loading**:
            - Loads the model with the specified architecture, backbone, and activation.
            - Loads pretrained weights from the checkpoint.
        3. **Evaluation Setup**:
            - Configures the loss function and metrics (Precision, Recall, IoU, F1).
        4. **Metrics Calculation**:
            - Iterates over the validation dataset.
            - Computes metrics for each class independently (multi-label evaluation).
            - Aggregates metrics across all validation samples.
        5. **Logging**:
            - Prints model and dataset details.
            - Prints class-specific metrics for the validation dataset.

    Example:
        >>> bonai_evaluate_multlabel_edns(
        ...     saved_ckpt="./saved_model.pth",
        ...     edn="Unet",
        ...     bb="vgg19",
        ...     data_type="segmentation",
        ...     classes=["footprint", "roof"],
        ...     loss_func="dice_loss"
        ... )

    Notes:
        - The metrics are normalized over the number of validation samples.
        - The model is evaluated in `eval` mode to disable gradient calculations.
    """
    ENCODER = bb
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENCODER_WEIGHTS = None
    ACTIVATION = 'sigmoid'

    transform = Compose([
        ToTensor(),  # Only keep ToTensor in transforms
    ])
    print("********************************************************************************")
    print("********************************************************************************")

    ####### DATASET GENERATOR
    DATA_DIR = './data/'+data_type+'/'
    x_train_dir, x_valid_dir, x_test_dir = os.path.join(DATA_DIR, 'train', 'image'), os.path.join(DATA_DIR, 'val', 'image'), os.path.join(DATA_DIR, 'test', 'image')
    y_train_dir, y_valid_dir, y_test_dir = os.path.join(DATA_DIR, 'train', 'label'), os.path.join(DATA_DIR, 'val', 'label'), os.path.join(DATA_DIR, 'test', 'label')
    # Dataset for train and val images
    #preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
    valid_dataset = BONAIDatasetFootprintRoof(img_dir=x_valid_dir, ann_dir=y_valid_dir, transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False) #num_workers=4


    ####### Load model, load pretrained Weights, and add into model
    model = get_model_from_smp(edn, bb, ENCODER_WEIGHTS, len(classes), ACTIVATION)
    model_checkpoint = torch.load(saved_ckpt)
    model_pretrained_state_dict = model_checkpoint['model_state_dict']
    model.load_state_dict(model_pretrained_state_dict)
    model.to(DEVICE)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())

    loss = get_loss(loss_func)
    metrics = [
        smp.utils.metrics.Precision(threshold=0.5),
        smp.utils.metrics.Recall(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5)
    ]

    ####### PRINTING SOME DETAILS
    print("Encoder: ", bb)
    print("Checkpoint: ", saved_ckpt)
    print("Validated on: ", data_type)
    print("Class: ", classes)
    print("Net Params: ", params)

    # Initialize dictionaries to store metrics for each class
    class_metrics = {cls: {metric.__name__: 0 for metric in metrics} for cls in classes}

    # Iterate over the validation dataset
    with torch.no_grad():
        for images, true_masks in valid_dataloader:
            images = images.to(DEVICE)
            true_masks = [mask.to(DEVICE) for mask in torch.unbind(true_masks, dim=1)]
            outputs = model(images)
            
            # Assuming `outputs` is of shape [batch_size, num_classes, height, width]
            for i, class_name in enumerate(classes):
                output_class = outputs[:, i, :, :]
                for metric in metrics:
                    class_metrics[class_name][metric.__name__] += metric(output_class, true_masks[i]).item()

    # Normalize metrics over the dataset
    num_samples = len(valid_dataloader)
    for class_name in class_metrics:
        for metric_name in class_metrics[class_name]:
            class_metrics[class_name][metric_name] /= num_samples

    # Print the results
    for class_name in class_metrics:
        print(f"Metrics for {class_name.capitalize()}:")
        for metric_name, metric_value in class_metrics[class_name].items():
            print(f"  {metric_name.capitalize()}: {metric_value:.4f}")

    print("********************************************************************************")
    print("********************************************************************************")




# Examples:

# Pre-training a model:
# *******************************
# vgg19
#train_unet_encoders('Unet', 'vgg19', 'WHU-256', ['building'], 16, 50, 0.0001, 'Adam', 'dice_loss', 'Pretrained-')
#train_unet_encoders('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 50, 0.0001, 'Adam', 'dice_loss', 'NoSDA-')

# BONAI dataset
#bonai_multilabel_training('Unet', 'vgg19', 'BONAI-shape', ['footprint','roof'], 8, 50, 0.0001, 'Adam', 'dice_loss', 'NoSDA-multilabel-')






# Normal fine-tuning upto any selected Block (without PGFiT or gradual unfreezing):
# *******************************
########## Load Pre-trained checkpoint
#pretrained_ckpt = "./trained_models/SDA-strategies/Pretrained-Unet-vgg19-WHU-256-50ep-Adam-dice_loss.pth"

########## Normal Fine-tuning by block name
#fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 50, 0.00001, 'Adam', 'dice_loss', pretrained_ckpt, 'decoder')
#fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 50, 0.00001, 'Adam', 'dice_loss', pretrained_ckpt, 'bottleneck')
#fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 50, 0.00001, 'Adam', 'dice_loss', pretrained_ckpt, 'Block 4')
#fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 50, 0.00001, 'Adam', 'dice_loss', pretrained_ckpt, 'Block 3')
#fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 50, 0.00001, 'Adam', 'dice_loss', pretrained_ckpt, 'Block 2')
#fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 50, 0.00001, 'Adam', 'dice_loss', pretrained_ckpt, 'Block 1')
#fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 50, 0.00001, 'Adam', 'dice_loss', pretrained_ckpt, 'Block 0')






########################## PGFiT - MELB Dataset ##########################
#gradual_fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 100, 0.0001, 'Adam', 'dice_loss', pretrained_ckpt, 'LR4')


######################### PGFiT - Massachusetts DATASET ##################
#gradual_fine_tune_unet_vgg19('Unet', 'vgg19', 'Massachusetts', ['building'], 16, 50, 0.0001, 'Adam', 'dice_loss', pretrained_ckpt, 'LR4')


######################### PGFiT - BONAI DATASET ##########################
#imagenet_ckpt = "./trained_models/SDA-strategies/unet_vgg19_imagenet.pth"
#bonai_multilabel_gradual_fine_tune_unet_vgg19('Unet', 'vgg19', 'BONAI-shape', ['footprint','roof'], 8, 50, 0.0001, 'Adam', 'dice_loss', imagenet_ckpt, 'multilabel')






########################## DiscLR ##########################
#gradual_fine_tune_disclr_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 100, 0.0001, 'Adam', 'dice_loss', pretrained_ckpt, 'LR4')


########################## STLR ##########################
#gradual_fine_tune_stlr_unet_vgg19('Unet', 'vgg19', 'MELB-multires', ['building'], 16, 100, 0.0001, 'Adam', 'dice_loss', pretrained_ckpt, 'LR4')


########### Evaluate models
#evaluate_edns(pretrained_ckpt, 'Unet', 'vgg19', 'WHU-256', 'dice_loss')

