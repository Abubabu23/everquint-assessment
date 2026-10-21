# CIFAR-10 Image Classification (PyTorch)

Small CNN from scratch + residual-block improvement, trained on Google Colab (T4 GPU).

## Project structure
- `cifar10_classification.ipynb` - full code and outputs (data, models, training, evaluation)
- `best_baseline.pth`, `best_resnet.pth` - best checkpoints (by validation accuracy)
- `requirements.txt` - dependencies

## Training setup
- Data: CIFAR-10, 45,000 train / 5,000 validation (fixed split, seed 42) / 10,000 test
- Preprocessing: normalization (CIFAR mean/std); train-only augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip
- Loader: torch DataLoader, batch size 128, num_workers=2, pin_memory=True
- Optimizer: Adam, lr 1e-3; scheduler: CosineAnnealingLR; 20 epochs
- Loss: CrossEntropyLoss; checkpoint saved on best validation accuracy
- Reproducibility: seeds set for random, numpy, torch

## Model design decisions
- Baseline: 3 conv stages (Conv + BatchNorm + ReLU, MaxPool, Dropout 0.3) and an MLP head with Dropout 0.5
- BatchNorm for stable, faster training; Dropout to reduce overfitting
- Test set used only once, for final evaluation (model selection uses validation set)

## Part 2: Residual block (Option A)
Each residual block: two 3x3 convs with BatchNorm and a skip connection (1x1 conv shortcut when shape changes).
ResCNN replaces the plain conv stages with residual blocks, using a similar parameter budget for a fair comparison.

| Model | Parameters | Test accuracy |
|---|---|---|
| Baseline CNN | 667,178 | 79.5% |
| ResCNN | 620,714 | 85.4% |

Residual connections improved accuracy by about 5.9 points with fewer parameters.

## Observations
- Most misclassifications are animals predicted as "cat" (dog, deer, bird, frog, horse), likely due to the low 32x32 resolution.
- Results are from a single run per model; small variation is expected across runs.

## How to run
1. Open the notebook in Google Colab, set Runtime to GPU (T4)
2. Install dependencies if needed: `pip install -r requirements.txt`
3. Run all cells in order (training takes about 5-10 minutes per model)