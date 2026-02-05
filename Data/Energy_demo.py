"""
Energy Efficiency Dataset - ANFIS Demo

This example demonstrates ANFIS for predicting heating/cooling loads
of buildings based on architectural features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.anfis import ANFIS


def load_and_prepare_data(filepath):
    """Load and split energy efficiency dataset."""
    data = pd.read_excel(filepath)
    print(f"Dataset shape: {data.shape}")
    print(f"\nFeatures: {data.columns[:8].tolist()}")
    print(f"Target: {data.columns[8]}")
    
    X = data.iloc[:, :8].values
    y = data.iloc[:, 8].values
    
    return X, y


def compare_configurations(X_train, y_train, X_val, y_val, X_test, y_test):
    """Compare different ANFIS configurations."""
    configs = [
        {"n_clusters": 20, "lr": 0.01, "name": "Small (20 clusters)"},
        {"n_clusters": 50, "lr": 0.01, "name": "Medium (50 clusters)"},
        {"n_clusters": 80, "lr": 0.005, "name": "Large (80 clusters)"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print('='*60)
        
        model = ANFIS(n_clusters=config['n_clusters'], X=X_train)
        model.fit(X_train, y_train, X_val, y_val, 
                 epochs=150, lr=config['lr'], patience=15)
        
        mse, rmse, mae, r2 = model.test(X_test, y_test, plot=False)
        results.append({
            'name': config['name'],
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
    
    # Display comparison
    print(f"\n{'='*60}")
    print("CONFIGURATION COMPARISON")
    print('='*60)
    print(f"{'Configuration':<25} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print('-'*60)
    for r in results:
        print(f"{r['name']:<25} {r['mse']:>8.3f} {r['rmse']:>8.3f} "
              f"{r['mae']:>8.3f} {r['r2']:>8.3f}")


def visualize_results(model, X_test, y_test):
    """Create detailed visualization of results."""
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plot
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     '--', color='red', linewidth=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predictions')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(alpha=0.3)
    
    # Residual plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(alpha=0.3)
    
    # Residual histogram
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(alpha=0.3)
    
    # Error over predictions
    abs_errors = np.abs(residuals)
    axes[1, 1].scatter(y_pred, abs_errors, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Absolute Error Distribution')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("ANFIS Energy Efficiency Prediction Demo")
    print("="*60)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "Example", "ENB2012_data.xlsx")
    X, y = load_and_prepare_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    print(f"\nData splits:")
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")
    
    # Train single model
    print(f"\n{'='*60}")
    print("Training ANFIS Model")
    print('='*60)
    
    model = ANFIS(n_clusters=50, X=X_train)
    model.fit(X_train, y_train, X_val, y_val, 
             epochs=200, lr=0.01, patience=15)
    
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print('='*60)
    model.test(X_test, y_test, plot=False)
    
    # Detailed visualization
    print("\nGenerating detailed visualizations...")
    visualize_results(model, X_test, y_test)
    
    # Compare configurations (optional)
    response = input("\nRun configuration comparison? (y/n): ")
    if response.lower() == 'y':
        compare_configurations(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == '__main__':
    main()