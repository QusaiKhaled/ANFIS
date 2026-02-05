# ANFIS: Theoretical Foundation and Implementation Details

## Overview

This document provides a comprehensive mathematical treatment of the Adaptive Neuro-Fuzzy Inference System (ANFIS) and explains the vectorization strategies employed in our implementation.

## 1. Theoretical Foundation

ANFIS combines neural networks' learning capability with fuzzy logic's reasoning power. The system approximates nonlinear functions through a collection of fuzzy if-then rules while adapting parameters through gradient descent and least squares estimation.

### 1.1 Fuzzy Inference Systems

A fuzzy inference system consists of:

- **Fuzzification**: Converting crisp inputs to fuzzy membership degrees
- **Rule Evaluation**: Computing firing strengths of fuzzy rules
- **Aggregation**: Combining rule outputs
- **Defuzzification**: Producing crisp output

### 1.2 Takagi-Sugeno Model

Our ANFIS implements the Takagi-Sugeno (TS) fuzzy model with rules of the form:

```
IF x₁ is A₁ AND x₂ is A₂ AND ... AND xF is AF 
THEN y = f(x₁, x₂, ..., xF)

```

where:

- Aᵢ are fuzzy sets defined by membership functions
- f is a crisp function (typically linear in TS models)

## 2. Network Architecture

Our ANFIS follows a 5-layer feedforward structure:

### Layer 1: Input Layer

- Transmits input values directly
- No computation performed
- Output: **x** = [x₁, x₂, ..., xF]

### Layer 2: Fuzzification Layer

- Applies Gaussian membership functions
- Each node represents a fuzzy set
- For cluster j and feature i:

```
μⱼᵢ(xᵢ) = exp(-(xᵢ - cⱼᵢ)² / (2σⱼᵢ²))

```

Parameters:

- cⱼᵢ: center (mean) of Gaussian
- σⱼᵢ: standard deviation (width)

### Layer 3: Product Layer (Implicit in Implementation)

- Computes firing strength of each rule
- Uses product t-norm (AND operation):

```
wⱼ(x) = ∏ᵢ₌₁ᶠ μⱼᵢ(xᵢ)

```

Due to exponential properties:

```
wⱼ(x) = exp(-∑ᵢ₌₁ᶠ (xᵢ - cⱼᵢ)² / (2σⱼᵢ²))

```

### Layer 4: Normalization Layer

- Normalizes firing strengths to sum to 1
- Ensures interpretability as fuzzy membership degrees:

```
w̄ⱼ(x) = wⱼ(x) / ∑ₖ₌₁ᴷ wₖ(x)

```

where K is the number of rules/clusters.

### Layer 5: Defuzzification Layer

- Weighted combination of rule outputs
- Linear consequent functions:

```
y(x) = ∑ⱼ₌₁ᴷ w̄ⱼ(x) · fⱼ

```

In our implementation, fⱼ = wⱼ (scalar weights).

## 3. Mathematical Formulation

### 3.1 Forward Pass

Given input **x** = [x₁, x₂, ..., xF]:

1. **Compute differences**: dⱼᵢ = xᵢ - cⱼᵢ
2. **Squared normalized distances**: sⱼᵢ = dⱼᵢ² / (2σⱼᵢ²)
3. **Firing strengths**: wⱼ = exp(-∑ᵢ sⱼᵢ)
4. **Normalized strengths**: w̄ⱼ = wⱼ / ∑ₖ wₖ
5. **Output**: y = ∑ⱼ w̄ⱼ · θⱼ

### 3.2 Parameter Sets

**Premise Parameters** (nonlinear):

- Centers: **C** = {cⱼᵢ} for j = 1..K, i = 1..F
- Standard deviations: **Σ** = {σⱼᵢ}

**Consequent Parameters** (linear):

- Weights: **θ** = [θ₁, θ₂, ..., θK]ᵀ

## 4. Hybrid Learning Algorithm

### 4.1 Least Squares Estimation (Consequent Parameters)

With premise parameters fixed, the output is linear in consequent parameters:

```
y = ∑ⱼ w̄ⱼ · θⱼ = **w̄ᵀθ**

```

For N training samples, we construct the system:

```
**Wθ** = **y**

```

where **W** is an N × K matrix of normalized firing strengths.

**Closed-form solution:**

```
θ = (**WᵀW**)⁻¹**Wᵀy**

```

In practice, we use numpy.linalg.lstsq which handles numerical stability.

### 4.2 Gradient Descent (Premise Parameters)

With consequent parameters fixed, we optimize premise parameters by minimizing mean squared error:

```
E = 1/(2N) ∑ₙ₌₁ᴺ (yₙ - ŷₙ)²

```

#### 4.2.1 Center Gradients

Chain rule application:

```
∂E/∂cⱼᵢ = -1/N ∑ₙ (yₙ - ŷₙ) · ∂ŷₙ/∂cⱼᵢ

```

Computing the derivative:

```
∂ŷₙ/∂cⱼᵢ = ∂ŷₙ/∂w̄ⱼ · ∂w̄ⱼ/∂wⱼ · ∂wⱼ/∂cⱼᵢ

```

Key derivatives:

- ∂ŷₙ/∂w̄ⱼ = θⱼ (consequent weight)
- ∂wⱼ/∂cⱼᵢ = wⱼ · (xᵢ - cⱼᵢ) / σⱼᵢ²

The normalization derivative ∂w̄ⱼ/∂wⱼ requires the quotient rule, but in our implementation we compute the full gradient directly through the membership function.

**Update rule:**

```
cⱼᵢ ← cⱼᵢ - η · ∂E/∂cⱼᵢ

```

#### 4.2.2 Standard Deviation Gradients

Similarly:

```
∂E/∂σⱼᵢ = -1/N ∑ₙ (yₙ - ŷₙ) · ∂ŷₙ/∂σⱼᵢ

```

Key derivative:

```
∂wⱼ/∂σⱼᵢ = wⱼ · (xᵢ - cⱼᵢ)² / σⱼᵢ³

```

**Update rule:**

```
σⱼᵢ ← max(σⱼᵢ - η · ∂E/∂σⱼᵢ, σₘᵢₙ)

```

The max operation ensures numerical stability by preventing division by zero.

### 4.3 Training Procedure

```
for epoch in 1 to max_epochs:
    # Forward pass
    W = compute_memberships(X)
    
    # Least squares (consequent)
    θ = solve(W, y)
    
    # Forward prediction
    ŷ = W @ θ
    
    # Compute error
    e = y - ŷ
    
    # Gradient descent (premise)
    ∇C = compute_center_gradients(X, e, W, θ)
    ∇Σ = compute_std_gradients(X, e, W, θ)
    
    C ← C - η · clip(∇C)
    Σ ← max(Σ - η · clip(∇Σ), σₘᵢₙ)
    
    # Early stopping check
    if validation_mse > best_mse:
        wait += 1
        if wait > patience:
            break

```

## 5. Vectorization Strategy

### 5.1 Broadcasting Mechanics

NumPy broadcasting allows operations on arrays of different shapes by automatically expanding dimensions.

**Example: Computing differences**

```python
# X: (N, F) - N samples, F features
# centers: (K, F) - K clusters, F features

# Reshape for broadcasting
X_expanded = X[:, None, :]      # (N, 1, F)
C_expanded = centers[None, :, :]  # (1, K, F)

# Broadcast subtraction
diffs = X_expanded - C_expanded   # (N, K, F)

```

Result: diffs[n, k, f] = X[n, f] - centers[k, f]

### 5.2 Key Vectorized Operations

#### 5.2.1 Membership Computation

```python
# Compute all memberships at once
diffs = X[:, None, :] - self.centers[None, :, :]  # (N, K, F)
sq = diffs**2 / (2 * self.stds[None, :, :]**2)   # (N, K, F)
M = np.exp(-sq.sum(axis=2))                        # (N, K)
M = M / M.sum(axis=1, keepdims=True)              # normalize

```

**Performance**: O(1) parallel operations vs O(N×K×F) sequential loops

#### 5.2.2 Gradient Computation

```python
# Error broadcasting
err = (y - y_pred)[:, None, None]  # (N, 1, 1)

# Reuse diffs from forward pass
dM_dc = M[:, :, None] * diffs / (self.stds[None, :, :]**2)
dY_dc = dM_dc * self.w[None, :, None]

# Aggregate over samples
grad_c = -2/N * (err * dY_dc).sum(axis=0)  # (K, F)

```

### 5.3 Memory Efficiency

**Tensor Reuse**: The (N, K, F) difference tensor is computed once and reused for:

1. Membership evaluation
2. Center gradients
3. Standard deviation gradients

**In-place Operations**: NumPy operations often work in-place when possible, reducing memory allocation overhead.

## 6. Computational Complexity


| Operation  | Loop-based          | Vectorized        |
| ---------- | ------------------- | ----------------- |
| Membership | O(N·K·F) sequential | O(N·K·F) parallel |
| Gradients  | O(N·K·F) sequential | O(N·K·F) parallel |
| Memory     | O(N·K·F)            | O(N·K·F)          |


**Speedup Factor**: Depends on hardware, typically:

- CPU (BLAS): 10-30x
- CPU (MKL): 30-50x
- GPU: 50-100x+

## 7. Numerical Stability

### 7.1 Gradient Clipping

```python
grad_c = np.clip(grad_c, -1, 1)
grad_s = np.clip(grad_s, -0.5, 0.5)

```

Prevents exploding gradients during training.

### 7.2 Minimum Standard Deviation

```python
self.stds = np.maximum(self.stds - lr * grad_s, 0.1)

```

Prevents division by zero in membership functions.

### 7.3 Least Squares Regularization

```python
theta = np.linalg.lstsq(M, y, rcond=None)[0]

```

The lstsq function handles rank-deficient cases automatically.

## 8. Initialization Strategy

### 8.1 K-Means Clustering

- Centers initialized at cluster centroids
- Provides good starting points in feature space
- Ensures rules cover the data distribution

### 8.2 Standard Deviations

- Initialized uniformly to 1.0
- Prevents over-fitting to initialization
- Adapted during training

## 9. Performance Considerations

### 9.1 Batch Processing

All samples processed simultaneously:

```python
M = self._membership(X)  # Vectorized over all N samples

```

### 9.2 Cache Efficiency

Sequential memory access patterns improve CPU cache utilization.

### 9.3 Parallelization

NumPy automatically leverages multi-core CPUs through BLAS libraries.

## 10. Extensions and Variations

### 10.1 Alternative Membership Functions

- Triangular: μ(x) = max(0, 1 - |x - c|/σ)
- Trapezoidal: Generalization of triangular
- Bell-shaped: μ(x) = 1/(1 + ((x-c)/σ)²ᵇ)

### 10.2 Different T-norms

- Minimum: wⱼ = min(μⱼ₁, μⱼ₂, ..., μⱼF)
- Product: wⱼ = ∏ᵢ μⱼᵢ (current implementation)
- Lukasiewicz: wⱼ = max(0, ∑ᵢ μⱼᵢ - F + 1)

### 10.3 Higher-order Consequents

Replace scalar weights with linear functions:

```
fⱼ = θⱼ₀ + θⱼ₁x₁ + θⱼ₂x₂ + ... + θⱼFxF

```

Increases model complexity but maintains linear optimization for consequent parameters.

## References

1. Jang, J.-S. R. (1993). "ANFIS: Adaptive-Network-Based Fuzzy Inference System". IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665-685.
2. Takagi, T., & Sugeno, M. (1985). "Fuzzy identification of systems and its applications to modeling and control". IEEE Transactions on Systems, Man, and Cybernetics, SMC-15(1), 116-132.
3. Pedrycz, W., & Gomide, F. (2007). "Fuzzy Systems Engineering: Toward Human-Centric Computing". John Wiley & Sons.
4. Harris, C. R., et al. (2020). "Array programming with NumPy". Nature, 585(7825), 357-362.

