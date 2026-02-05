# Quick Start Guide

## Installation

### Option 1: Clone and Install

```bash
git clone https://github.com/yourusername/vectorized-anfis.git
cd vectorized-anfis
pip install -r requirements.txt

```

### Option 2: Install as Package

```bash
git clone https://github.com/yourusername/vectorized-anfis.git
cd vectorized-anfis
pip install -e .

```

## Basic Usage

### 1. Simple Example

```python
from src.anfis import ANFIS
import numpy as np

# Generate synthetic data
X = np.random.randn(500, 3)
y = X[:, 0]**2 + X[:, 1] + 0.5*X[:, 2] + np.random.randn(500)*0.1

# Split data
split = 400
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create validation set
val_split = 360
X_train, X_val = X_train[:val_split], X_train[val_split:]
y_train, y_val = y_train[:val_split], y_train[val_split:]

# Initialize and train
model = ANFIS(n_clusters=20, X=X_train)
model.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.01)

# Predict
predictions = model.predict(X_test)

```

### 2. Energy Efficiency Example

```bash
# Make sure you have the data file
python main.py

```

Or run the detailed demo:

```bash
cd examples
python energy_demo.py

```

## Parameter Guide

### ANFIS Constructor

- **n_clusters**: Number of fuzzy rules (10-100 typical)
  - Small datasets: 10-30
  - Medium datasets: 30-50
  - Large datasets: 50-100+
- **X**: Training data for k-means initialization

### Training Parameters

- **epochs**: Maximum iterations (100-300 typical)
- **lr**: Learning rate (0.001-0.01 typical)
  - Start with 0.01 for small models
  - Use 0.001-0.005 for large models
- **patience**: Early stopping patience (10-20 typical)

## Performance Tips

### 1. Number of Clusters

- More clusters = higher capacity but slower training
- Use cross-validation to find optimal number
- Start with K = sqrt(N) as a rule of thumb

### 2. Learning Rate

- Too high: unstable training, oscillations
- Too low: slow convergence
- Monitor validation loss for guidance

### 3. Data Preprocessing

- **Always standardize features** (mean=0, std=1)
- Remove outliers if present
- Handle missing values before training

### 4. Early Stopping

- Essential for preventing overfitting
- Typical patience: 15-20 epochs
- Monitor validation MSE, not training MSE

## Common Issues

### Problem: Model not converging

**Solution:**

- Reduce learning rate (try 0.001)
- Increase patience (try 30-40)
- Check data scaling
- Try different n_clusters

### Problem: Overfitting

**Solution:**

- Reduce n_clusters
- Increase early stopping patience
- Add more training data
- Increase validation set size

### Problem: Training too slow

**Solution:**

- Reduce n_clusters
- Reduce max epochs
- Use smaller learning rate for faster convergence
- Ensure NumPy is using optimized BLAS

### Problem: Poor predictions

**Solution:**

- Check data preprocessing
- Verify feature scaling
- Try different n_clusters
- Increase training epochs
- Check for data leakage

## Example Output

```
Epoch 1, Train MSE: 15.3421, Val MSE: 15.8932
Epoch 2, Train MSE: 10.2341, Val MSE: 10.7654
Epoch 3, Train MSE: 6.7823, Val MSE: 7.1234
...
Epoch 45, Train MSE: 2.3421, Val MSE: 2.4567
Early stopping at epoch 45, best Val MSE: 2.4432

Test MSE: 2.5123, RMSE: 1.5850, MAE: 1.2341, R2: 0.9823

```

## Next Steps

- Read the [detailed theory](https://claude.ai/chat/docs/theory.md)
- Explore [examples](https://claude.ai/chat/examples/)
- Check the [API documentation](https://claude.ai/chat/f5bd78a9-5c0d-443e-842e-ddaaee6db528#api-reference)
- Try on your own dataset

## Getting Help

If you encounter issues:

1. Check this guide first
2. Review the examples in `examples/`
3. Read the theory documentation
4. Open an issue on GitHub

