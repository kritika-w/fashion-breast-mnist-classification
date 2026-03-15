
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
