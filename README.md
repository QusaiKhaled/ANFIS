# Data Directory

## Energy Efficiency Dataset

### Overview

The Energy Efficiency dataset contains 768 samples of building simulations used to assess heating and cooling load requirements based on building characteristics.

### Source

UCI Machine Learning Repository  
[https://archive.ics.uci.edu/ml/datasets/Energy+efficiency](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)

### File

- `ENB2012_data.xlsx` - Energy efficiency data (not included in repo)

### Download Instructions

Download the dataset from the UCI repository or prepare your own data file with the same structure.

### Features (8 input variables)

1. **X1**: Relative Compactness (0.62 - 0.98)
2. **X2**: Surface Area (514.5 - 808.5 m²)
3. **X3**: Wall Area (245.0 - 416.5 m²)
4. **X4**: Roof Area (110.25 - 220.5 m²)
5. **X5**: Overall Height (3.5 - 7.0 m)
6. **X6**: Orientation (2, 3, 4, 5 - categorical)
7. **X7**: Glazing Area (0 - 0.4, proportion)
8. **X8**: Glazing Area Distribution (0-5 - categorical)

### Targets (2 output variables)

- **Y1**: Heating Load (6.01 - 43.1 kWh/m²)
- **Y2**: Cooling Load (10.9 - 48.03 kWh/m²)

**Note**: Our examples use Y1 (Heating Load) as the target.

### Dataset Characteristics

- **Samples**: 768
- **Features**: 8
- **Task**: Regression
- **Missing Values**: None
- **Feature Types**: Mixed (continuous and categorical)

### Usage in ANFIS

The dataset is ideal for ANFIS because:

- Moderate size (not too small, not too large)
- Mix of continuous and categorical features
- Nonlinear relationships between inputs and output
- Real-world application (building energy efficiency)

### Data Preprocessing

When using this dataset:

1. **Standardization**: Apply StandardScaler to all features
2. **Train-Test Split**: Typical 80-20 split
3. **Validation Set**: 10% of training for early stopping
4. **No missing values**: Dataset is complete

### Expected Results

With proper hyperparameters:

- **MSE**: 2-4
- **RMSE**: 1.4-2.0
- **R²**: 0.97-0.99
- **MAE**: 1.0-1.5

### Citation

If you use this dataset, please cite:

```
A. Tsanas, A. Xifara: 'Accurate quantitative estimation of energy 
performance of residential buildings using statistical machine learning tools', 
Energy and Buildings, Vol. 49, pp. 560-567, 2012

```

### Alternative Datasets

ANFIS works well with various regression datasets:

- California Housing
- Boston Housing (deprecated but available)
- Concrete Compressive Strength
- Wine Quality
- Any continuous regression problem with 2-20 features

### Preparing Your Own Data

To use ANFIS with your data:

1. **Format**: CSV, Excel, or NumPy arrays
2. **Structure**: Features in columns, samples in rows
3. **Target**: Single continuous variable
4. **Size**: At least 100 samples recommended
5. **Preprocessing**: Handle missing values, outliers

Example loading custom data:

```python
import pandas as pd

# CSV file
data = pd.read_csv('your_data.csv')
X = data.iloc[:, :-1].values  # All columns except last
y = data.iloc[:, -1].values   # Last column

# Or NumPy arrays directly
X = np.load('features.npy')
y = np.load('targets.npy')

```

