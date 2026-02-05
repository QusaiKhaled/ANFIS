import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.anfis import ANFIS


def main():
    """Main execution function for ANFIS energy efficiency prediction."""
    
    # Load data
    data = pd.read_excel("Example/ENB2012_data.xlsx")
    X = data.iloc[:, :8].values
    y = data.iloc[:, 8].values
    
    # Train-test split
    Xt, Xs, yt, ys = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    sc = StandardScaler().fit(Xt)
    Xt, Xs = sc.transform(Xt), sc.transform(Xs)
    
    # Create validation set
    Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=0.1, random_state=42)
    
    # Initialize and train ANFIS
    print("Initializing ANFIS with 50 clusters...")
    model = ANFIS(n_clusters=50, X=Xtr)
    
    print("\nTraining ANFIS...")
    model.fit(Xtr, ytr, Xv, yv, epochs=200, lr=0.01, patience=15)
    
    print("\nEvaluating on test set...")
    model.test(Xs, ys)


if __name__ == '__main__':
    main()