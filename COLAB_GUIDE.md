# Gaussian Prototypical Networks - Google Colab Guide

## Quick Start for Google Colab

### Step 1: Clone the Repository
```python
!git clone https://github.com/Mohan6148/omniglot.git
%cd omniglot/gaussian-prototypical-networks
```

### Step 2: Install Dependencies
```python
!pip install -q -r requirements.txt
```

### Step 3: Setup Data and Directories
```python
!python colab_setup.py
```

### Step 4: Run Training
```python
!python classifier7.py
```

## What to Expect

The default configuration trains on a **small subset** (400 images) for quick testing:
- **Training**: 30 epochs on 60-way classification
- **Validation**: 20-way, 1-shot to 19-shot classification
- **Encoder**: Small CNN (64-dimensional embeddings)
- **Expected time**: ~5-10 minutes on Colab GPU

### Initial Output
You should see:
```
X_train (1600, 28, 28)
X_val (1600, 28, 28)
```

### Training Progress
```
# epoch=0 it=0/1 loss=4.094 train acc=0.017
Epoch = 0 done: loss = 4.094 train acc = 0.017
```

### Validation Results
```
Validation:
Epoch = 0 shot = 1 val acc = 0.05 +- 0.02
Epoch = 0 shot = 5 val acc = 0.15 +- 0.03
```

## For Full-Scale Training

To train on the complete Omniglot dataset, modify `classifier7.py`:

```python
# Line 14-16, change from:
labels_train, images_train, info_train, labels_val, images_val, info_val = loadOmniglot(
    path = "data/", train = 1, limit = 400
)

# To:
labels_train, images_train, info_train, labels_val, images_val, info_val = loadOmniglot(
    path = "data/", train = 0, limit = None
)
```

## Troubleshooting

### Out of Memory
If you run out of memory, reduce batch size:
- Decrease `N_classes` (line 74)
- Decrease `N_query` (line 77)

### Slow Training
Enable GPU in Colab:
- Runtime → Change runtime type → GPU

### Import Errors
Make sure you're in the correct directory:
```python
%cd /content/omniglot/gaussian-prototypical-networks
```

## Output Files

- **Checkpoints**: `model_checkpoints/name_of_the_run_e{epoch}_i{iter}.ckpt`
- **Visualizations**: `tmp_imgs/name_of_the_run_*.png`
- **Final plot**: `clustering_true.png` and `clustering_pred.png`

## TensorFlow Compatibility

This codebase uses **TensorFlow 1.x compatibility mode** to run in TensorFlow 2.x:
- All TF 1.x APIs are accessed via `tensorflow.compat.v1`
- Graph mode is enabled (eager execution disabled)
- No code changes needed from your side!
