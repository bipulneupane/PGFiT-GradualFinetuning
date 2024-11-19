# PGFiT - Phase-wise Gradual Fine-Tuning

Welcome to **PGFiT**, a repository dedicated to phase-wise gradual fine-tuning of encoder-decoder networks for segmentation tasks. This repository supports training and evaluation on single-label and multi-label datasets and provides implementations for various fine-tuning strategies.

---

## Supported Datasets

The repository is built around four datasets:
- **WHU Building Dataset**
- **Melbourne Building Dataset**
- **Massachusetts Building Dataset**
- **BONAI Dataset**

---


## Examples for U-VGG19
Refer to `train-vgg19.py` for more examples and details of the codes.


```python
# Pre-training examples
train_unet_encoders('Unet', 'vgg19', 'WHU', ['building'], 16, 50, 0.0001, 'Adam', 'dice_loss', 'Pretrained-')
train_unet_encoders('Unet', 'vgg19', 'MELB', ['building'], 16, 50, 0.0001, 'Adam', 'dice_loss', 'NoSDA-')
bonai_multilabel_training('Unet', 'vgg19', 'BONAI-shape', ['footprint', 'roof'], 8, 50, 0.0001, 'Adam', 'dice_loss', 'NoSDA-multilabel-')

# Normal Fine-Tuning Without Gradual Unfreezing
fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB', ['building'], 16, 50, 0.00001, 'Adam', 'dice_loss', pretrained_ckpt, 'decoder')

# Proposed PGFiT (examples for MELB dataset)
gradual_fine_tune_unet_vgg19('Unet', 'vgg19', 'MELB', ['building'], 16, 100, 0.0001, 'Adam', 'dice_loss', pretrained_ckpt, 'LR4')

# Gradual Fine-tuning with Discriminative Learning Rate (DiscLR)
gradual_fine_tune_disclr_unet_vgg19('Unet', 'vgg19', 'MELB', ['building'], 16, 100, 0.0001, 'Adam', 'dice_loss', pretrained_ckpt, 'LR4')

# Gradual Fine-tuning with Slanted Triangular Learning Rate (STLR)
gradual_fine_tune_stlr_unet_vgg19('Unet', 'vgg19', 'MELB', ['building'], 16, 100, 0.0001, 'Adam', 'dice_loss', pretrained_ckpt, 'LR4')

# Evaluate Model
evaluate_edns(pretrained_ckpt, 'Unet', 'vgg19', 'WHU', 'dice_loss')
```
---


## Examples for U-MiTb2

Refer to train-mitb2.py for more examples and details of the codes.

```python
# Pre-training a Model
train_unet_encoders('Unet', 'mit_b2', 'WHU', ['building'], 16, 50, 0.0001, 'Adam', 'dice_loss', 'Pretrained-')

# Proposed PGFiT (examples for BONAI dataset)
imagenet_ckpt = "./trained_models/SDA-strategies/unet_mitb2_imagenet.pth"
bonai_multilabel_gradual_fine_tune_unet_mit_b2('Unet', 'mit_b2', 'BONAI-shape', ['footprint', 'roof'], 8, 50, 0.0001, 'Adam', 'dice_loss', imagenet_ckpt, 'multilabel')

# Gradual Fine-tuning with Discriminative Learning Rate (DiscLR)
gradual_fine_tune_disclr_unet_mit_b2('Unet', 'mit_b2', 'MELB', ['building'], 16, 50, 0.0001, 'Adam', 'dice_loss', pretrained_ckpt, 'LR4')

# Gradual Fine-tuning with Slanted Triangular Learning Rate (STLR)
gradual_fine_tune_stlr_unet_mit_b2('Unet', 'mit_b2', 'MELB', ['building'], 16, 100, 0.0001, 'Adam', 'dice_loss', pretrained_ckpt, 'LR4')

# Evaluate Model
evaluate_edns(pretrained_ckpt, 'Unet', 'mit_b2', 'MELB', 'dice_loss')
```

---

## Required Python Libraries and Packages

The following Python packages are required to run the code in this repository:

```
os
torch
numpy
pillow
torchvision
cv2
matplotlib
json
shapely
segmentation_models_pytorch (https://github.com/qubvel-org/segmentation_models.pytorch)
albumentations (https://pypi.org/project/albumentations/0.0.10/)
```

---

## Collecting the Datasets

WHU Building Dataset (Aerial imagery dataset): [http://gpcv.whu.edu.cn/data/building_dataset.html](http://gpcv.whu.edu.cn/data/building_dataset.html)

Melbourne Building dataset: [https://arxiv.org/abs/2303.09064v4](https://arxiv.org/abs/2303.09064v4)

Massachusetts Building Dataset: [https://www.cs.toronto.edu/~vmnih/data/](https://www.cs.toronto.edu/~vmnih/data/)

BONAI Dataset: [https://github.com/jwwangchn/BONAI](https://github.com/jwwangchn/BONAI)

---

## Credit and Citation

This work is currently under revision in an IEEE journal. The appropriate citation will be provided once the paper is published.

---

## Contact

This repo is maintained by Bipul Neupane.
Work email: bneupane@student.unimelb.edu.au
Personal email: geomat.bipul@gmail.com

---

## Acknowledgements

The authors would like to acknowledge the creators of the datasets. These datasets has been invaluable for advancing research in off-nadir aerial image segmentation. The authors also acknowledge the contributors of the SegmentationModelsPytorch library, TIMM database, and HuggingFace for providing a comprehensive Python package to train CNNs and ViTs.

---




