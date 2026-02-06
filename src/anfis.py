import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


class ANFIS:
    """Vectorized ANFIS implementation using NumPy broadcasting for high performance.
    
    Combines fuzzy logic reasoning with neural network learning through a 5-layer
    architecture optimized with vectorized operations.
    """
    
    def __init__(self, n_clusters, X):
        """Initialize ANFIS with k-means clustering.
        
        Args:
            n_clusters: Number of fuzzy rules/clusters
            X: Training data of shape (N, F) for initialization
        """
        self.n_clusters, self.n_features = n_clusters, X.shape[1]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        self.centers = kmeans.cluster_centers_  # shape (K, F)
        self.stds = np.full((n_clusters, X.shape[1]), 1)  # shape (K, F)

    def _membership(self, X):
        """Compute normalized fuzzy membership degrees using Gaussian functions.
        
        Args:
            X: Input data of shape (N, F)
            
        Returns:
            Normalized membership matrix of shape (N, K)
        """
        diffs = X[:, None, :] - self.centers[None, :, :]  # broadcast diff
        sq = diffs**2 / (2 * self.stds[None, :, :]**2)  # distance
        M = np.exp(-sq.sum(axis=2))  # sum over features -> (N, K)
        return M / M.sum(axis=1, keepdims=True)  # normalize memberships

    def fit(self, X, y, Xv, yv, epochs=100, lr=1e-4, patience=20):
        """Train ANFIS using hybrid learning (least squares + gradient descent).
        
        Args:
            X: Training features of shape (N, F)
            y: Training targets of shape (N,)
            Xv: Validation features
            yv: Validation targets
            epochs: Maximum training iterations
            lr: Learning rate for gradient descent
            patience: Early stopping patience
        """
        N = X.shape[0]
        best_mse, wait = np.inf, 0
        
        for ep in range(epochs):
            # Least squares for consequent parameters
            M = self._membership(X)  # (N, K)
            theta = np.linalg.lstsq(M, y, rcond=None)[0]
            self.w = theta
            y_pred = M @ self.w
            mse = mean_squared_error(y, y_pred)
            
            # Validation
            Mv = self._membership(Xv)
            val_mse = mean_squared_error(yv, Mv @ self.w)
            print(f"Epoch {ep+1}, Train MSE: {mse:.4f}, Val MSE: {val_mse:.4f}", flush=True)
            
            # Gradient descent for premise parameters
            err = (y - y_pred)[:, None, None]  # (N, 1, 1)
            diffs = X[:, None, :] - self.centers[None, :, :]
            dM_dc = M[:, :, None] * diffs / (self.stds[None, :, :]**2)
            dM_ds = M[:, :, None] * diffs**2 / (self.stds[None, :, :]**3)
            dY_dc = dM_dc * self.w[None, :, None]
            dY_ds = dM_ds * self.w[None, :, None]
            grad_c = -2/N * (err * dY_dc).sum(axis=0)
            grad_s = -2/N * (err * dY_ds).sum(axis=0)
            grad_c = np.clip(grad_c, -1, 1)
            grad_s = np.clip(grad_s, -0.5, 0.5)
            
            self.centers -= lr * grad_c
            self.stds = np.maximum(self.stds - lr * grad_s, 0.1)
            
            # Early stopping
            if val_mse < best_mse:
                best_mse, wait = val_mse, 0
            else:
                wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep+1}, best Val MSE: {best_mse:.4f}", flush=True)
                break

    def predict(self, X):
        """Generate predictions for input data.
        
        Args:
            X: Input features of shape (N, F)
            
        Returns:
            Predictions of shape (N,)
        """
        M = self._membership(X)
        return M @ self.w

    def test(self, X, y, plot=True):
        """Evaluate model performance on test data.
        
        Args:
            X: Test features
            y: Test targets
            plot: Whether to display scatter plot
            
        Returns:
            Tuple of (MSE, RMSE, MAE, R2)
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}", flush=True)
        
        if plot:
            plt.figure(figsize=(5, 4))
            plt.scatter(y, y_pred, alpha=0.6)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
            plt.title('Actual vs Predicted')
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.tight_layout()
            plt.show()
        
        return mse, rmse, mae, r2