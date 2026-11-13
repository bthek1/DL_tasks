# Deep Learning Equations

Deep Learning Equations

Author

Benedict Thekkel

## 1. **The Basics: Neural Network as a Function**

At its core, a **neural network** is a function $f(\mathbf{x}; \mathbf{\theta})$\
- $\mathbf{x}$ = input vector\
- $\mathbf{\theta}$ = model parameters (weights and biases)

It **maps inputs to outputs** by applying a series of **linear transformations** and **non-linear activations**.

------------------------------------------------------------------------

## 2. **Linear Transformation (Affine Function)**

Each **layer** in a neural network computes: $$\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b}$$\
- $\mathbf{W}$ = weight matrix\
- $\mathbf{b}$ = bias vector\
- $\mathbf{z}$ = pre-activation output

### Example:

For a single neuron:\
$$z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

------------------------------------------------------------------------

## 3. **Activation Functions**

Activation functions introduce **non-linearity**.

| Function | Equation | Purpose |
|----|----|----|
| Sigmoid | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | Outputs between $0$ and $1$ |
| Tanh | $\tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ | Outputs between $-1$ and $1$ |
| ReLU | $\text{ReLU}(z) = \max(0, z)$ | Introduces sparsity |
| Leaky ReLU | $\text{LeakyReLU}(z) = \max(\alpha z, z)$ | Avoids dying neurons |
| Softmax | $\text{Softmax}(z_i) = \dfrac{e^{z_i}}{\sum_{j} e^{z_j}}$ | Outputs probabilities |

------------------------------------------------------------------------

## 4. **Forward Propagation (The Core Computation)**

For a **deep neural network** with $L$ layers: $$\mathbf{a}^{[0]} = \mathbf{x} \\
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \\
\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]}) \\
\text{For } l = 1, 2, \dots, L$$\
- $\mathbf{a}^{[l]}$ = activation of layer $l$\
- $g^{[l]}$ = activation function for layer $l$

### Final output:

$$\hat{\mathbf{y}} = \mathbf{a}^{[L]}$$

------------------------------------------------------------------------

## 5. **Loss Functions (How We Measure Error)**

| Task | Loss Function | Equation |
|----|----|----|
| Regression | Mean Squared Error (MSE) | $\mathcal{L} = \dfrac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$ |
| Binary Classification | Binary Cross-Entropy (Log Loss) | $\mathcal{L} = - \dfrac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$ |
| Multi-class Classification | Categorical Cross-Entropy | $\mathcal{L} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)$ |

------------------------------------------------------------------------

## 6. **Backpropagation (How We Learn)**

It’s **gradient descent** applied via the **chain rule**.

### 6.1 Compute Gradients:

$$\dfrac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \delta^{[l]} (\mathbf{a}^{[l-1]})^T \\
\dfrac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \delta^{[l]}$$

### 6.2 Backpropagate Errors:

$$\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot g'(\mathbf{z}^{[l]})$$\
- $\delta^{[l]}$ = error at layer $l$\
- $\odot$ = element-wise multiplication

------------------------------------------------------------------------

## 7. **Gradient Descent (Parameter Updates)**

For each parameter $\theta$ (weights or biases):\
$$\theta := \theta - \alpha \dfrac{\partial \mathcal{L}}{\partial \theta}$$\
- $\alpha$ = learning rate

------------------------------------------------------------------------

## 8. **Optimizers (Variants of Gradient Descent)**

[TABLE]

------------------------------------------------------------------------

## 9. **Regularization (Prevent Overfitting)**

- **L2 Regularization** (Weight Decay):\
  $$\mathcal{L}_{\text{reg}} = \mathcal{L} + \dfrac{\lambda}{2m} \sum_{l} \|\mathbf{W}^{[l]}\|^2$$
- **Dropout**:\
  $$\mathbf{a}^{[l]} = \mathbf{a}^{[l]} \odot \mathbf{d}^{[l]}$$\
- $\mathbf{d}^{[l]}$ = dropout mask

------------------------------------------------------------------------

## 10. **Convolution (CNNs)**

$$(\mathbf{I} * \mathbf{K})(i,j) = \sum_m \sum_n \mathbf{I}(i + m, j + n) \cdot \mathbf{K}(m, n)$$\
- $\mathbf{I}$ = input image\
- $\mathbf{K}$ = kernel (filter)

------------------------------------------------------------------------

## 11. **Recurrent Neural Networks (RNNs)**

### Hidden State Update:

$$\mathbf{h}_t = g(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$\
\### Output:\
$$\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

------------------------------------------------------------------------

## 12. **Attention Mechanism (Transformers)**

### Self-Attention Equation:

$$\text{Attention}(Q, K, V) = \text{softmax} \left( \dfrac{QK^T}{\sqrt{d_k}} \right) V$$\
- $Q$ = query\
- $K$ = key\
- $V$ = value\
- $d_k$ = dimension of $K$

------------------------------------------------------------------------

## 13. **Transformer Feedforward Layer**

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

------------------------------------------------------------------------

## 14. **Training Objective (Put it all together!)**

### Find $\mathbf{\theta}$ that minimizes the loss:

$$\mathbf{\theta}^* = \underset{\mathbf{\theta}}{\arg\min} \; \mathcal{L}(f(\mathbf{x}; \mathbf{\theta}), \mathbf{y})$$

------------------------------------------------------------------------

## 🌍 **Forward-Looking Concepts**

| Concept | Equation | Purpose |
|----|----|----|
| Contrastive Loss (CLIP) | $\mathcal{L} = -\log \dfrac{\exp(\text{sim}(\mathbf{x}, \mathbf{y})/\tau)}{\sum_{i=1}^{N} \exp(\text{sim}(\mathbf{x}, \mathbf{y}_i)/\tau)}$ | Align different modalities (text, image) |
| Diffusion Models | $\mathbf{x}_{t-1} = \dfrac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \dfrac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$ | Generate high-quality images |
| Reinforcement Learning | $Q(s,a) = r + \gamma \max_{a'} Q(s', a')$ | Optimize sequential decision-making |

------------------------------------------------------------------------

Back to top
