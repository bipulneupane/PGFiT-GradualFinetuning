# Dataset Directory Structure

Place the datasets in this directory following the folder structure outlined below. This structure is required for the training and evaluation scripts to function correctly.

## Example Folder Structure

- WHU/
    ├── train/
    │   ├── image/
    │   └── label/
    ├── val/
    │   ├── image/
    │   └── label/
    └── test/
        ├── image/
        └── label/


## Folder Descriptions

- **`train/`**: Contains training images and their corresponding labels.
  - **`image/`**: Directory for input training images.
  - **`label/`**: Directory for ground truth labels for training images.
- **`val/`**: Contains validation images and their corresponding labels.
  - **`image/`**: Directory for input validation images.
  - **`label/`**: Directory for ground truth labels for validation images.
- **`test/`**: Contains test images and their corresponding labels.
  - **`image/`**: Directory for input test images.
  - **`label/`**: Directory for ground truth labels for test images.

## Notes
- Ensure the folder names are **consistent** (e.g., `train`, `val`, `test`, `image`, `label`) as the scripts rely on this naming convention.
- Images and labels must be **correctly paired** within their respective folders.

This structure ensures seamless integration with the dataset loader in the provided codebase.