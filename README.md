
# fashion and breast mnist classification with deep learning

## overview
this project implements two neural network architectures for image classification tasks:
1. multi-layer perceptron (mlp) for fashionmnist clothing classification
2. lenet-5 convolutional neural network for breast cancer detection from breastmnist ultrasound images

## key features
- custom mlp implementation with dropout layers for regularization
- lenet-5 cnn architecture adapted for medical image classification
- data augmentation (random flips, rotations) applied to training set
- model checkpointing based on validation performance
- comprehensive evaluation metrics including accuracy, precision, recall, specificity, and auroc
- training/validation curves visualization
- clinical impact analysis for breast cancer screening application

## results

### fashionmnist (mlp)
- test accuracy: ~88% (final model based on best validation checkpoint)

### breastmnist (lenet)
| metric | value |
|--------|-------|
| accuracy | 82.69% |
| precision | 83.73% |
| recall (sensitivity) | 94.74% |
| specificity | 50.00% |
| negative predictive value | 77.78% |
| auroc | 0.8300 |

note: results may vary slightly between runs due to random initialization and data splits.

## project structure
├── assignment_01_code.ipynb # main notebook with all code
├── assignment 1 report-1.pdf # detailed analysis and discussion
├── requirements.txt # dependencies
└── README.md # this file


## requirements
- python 3.8+
- pytorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- medmnist


## model architectures
### mlp for fashionmnist
| layer | type | input | output | activation |
|-------|------|-------|--------|------------|
| 1 | linear | 28x28 | 256 | relu |
| 2 | linear | 256 | 128 | relu |
| 3 | dropout | 128 | 128 | - |
| 4 | linear | 128 | 64 | relu |
| 5 | dropout | 64 | 64 | - |
| 6 | linear | 64 | 10 | - |

### lenet for breastmnist
| layer | type | out channels | kernel | activation |
|-------|------|--------------|--------|------------|
| 1 | conv2d | 6 | 5x5 | relu |
| 2 | maxpool | 6 | 2x2 | - |
| 3 | conv2d | 16 | 5x5 | relu |
| 4 | maxpool | 16 | 2x2 | - |
| 5 | conv2d | 120 | 5x5 | relu |
| 6 | linear | 84 | - | relu |
| 7 | linear | 2 | - | - |
