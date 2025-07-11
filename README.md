# Brain-Tumor-Detection

This project uses deep learning and computer vision to detect brain tumors from MRI images. It leverages a fine-tuned VGG16 architecture for high-accuracy classification.

## Model Overview

The model is built on top of VGG16, a pre-trained convolutional neural network (CNN) originally trained on the ImageNet dataset.

### Key Features:
- Input images resized to **128x128**.
- **VGG16** used with `include_top=False` and `weights='imagenet'`.
- All base VGG layers are **frozen** except the last 3.
- **Flatten → Dropout → Dense → Dropout → Dense(softmax)** sequence added.
- Final output layer uses **softmax** for multiclass classification (e.g., glioma, meningioma, pituitary, notumor).

### Regularization:
- **Dropout layers (0.3 and 0.2)** are used to prevent overfitting.
- **Relu** activation is used in the intermediate dense layer.

---

> This README is auto-merged from local and remote sources during Git synchronization.