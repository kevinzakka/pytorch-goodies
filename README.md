## Model Statistics

### Number of Parameters

```python
num_params = sum(p.numel() for p in model.parameters()) # Total parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable parameters
```

### Number of FLOPS

[...]

## Weight Initialization

PyTorch layers are initialized by default in their respective `reset_parameters()` method. For example:

- `nn.Linear`
    - `weight` and `bias`: uniform distribution [-limit, +limit] where `limit` is `1. / sqrt(fan_in)` and `fan_in` is the number of input units in the weight tensor.
- `nn.Conv2D`
    - `weight` and `bias`: uniform distribution [-limit, +limit] where `limit` is `1. / sqrt(fan_in)` and `fan_in` is the number of input units in the weight tensor.

With this implementation, the variance of the layer outputs is equal to `Var(W) = 1 / 3 * sqrt(fan_in)` which isn't the best initialization strategy out there.

Note that PyTorch provides convenience functions for some of the initializations. The input and output shapes are computed using the method `_calculate_fan_in_and_fan_out()` and a `gain()` method scales the standard deviation to suit a particular activation.

### Xavier Initialization

This initialization is general-purpose and meant to "work" pretty well for any activation in practice.

```python
# default xavier init
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform(m.weight)
```

You can tailor this initialization to your specific activation by using the `nn.init.calculate_gain(act)` argument.

```python
# default xavier init
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
```

- [arXiv](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

### He et. al Initialization

This is a similarly derived initialization tailored specifically for ReLU activations since they do not exhibit zero mean.

```python
# he initialization
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal(m.weight, mode='fan_in')
```

For `mode=fan_in`, the variance of the distribution is ensured in the forward pass, while for `mode=fan_out`, it is ensured in the backwards pass.

- [arXiv](https://arxiv.org/abs/1502.01852)

### SELU Initialization

Again, this initialization is specifically derived for the SELU activation function. The authors use the `fan_in` strategy. They mention that there is no significant difference between sampling from a Gaussian, a truncated Gaussian or a Uniform distribution.

```python
# selu init
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
    elif isinstance(m, nn.Linear):
        fan_in = m.in_features
        nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
```

- [arXiv](https://arxiv.org/abs/1706.02515)

### Orthogonal Initialization

Orthogonality is a desirable quality in NN weights in part because it is norm preserving, i.e. it rotates the input matrix, but cannot change its norm (scale/shear). This property is valuable in deep or recurrent networks, where repeated matrix multiplication can result in signals vanishing or exploding.

```python
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal(m.weight)
```

- [arXiv](https://arxiv.org/abs/1312.6120)
- [Blog Post](https://hjweide.github.io/orthogonal-initialization-in-convolutional-layers)
- [Smerity Blog Post](https://smerity.com/articles/2016/orthogonal_init.html)
- [Google+ Discussion](https://plus.google.com/+SoumithChintala/posts/RZfdrRQWL6u)
- [Reddit Discussion](https://www.reddit.com/r/MachineLearning/comments/2qsje7/how_do_you_initialize_your_neural_network_weights/)

### Batch Norm Initialization

```python
for m in model:
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
```

## Weight Regularization

### L2 Regularization

Heavily penalizes peaky weight vectors and encourages diffuse weight vectors. Has the appealing property of encouraging the network to use all of its inputs a little rather that some of its inputs a lot.

```python
with torch.enable_grad():
    reg = 1e-6
    l2_loss = torch.zeros(1)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5 * reg * torch.sum(torch.pow(W, 2)))
```

### L1 Regularization

Encourages sparsity, meaning we encourage the network to select the most useful inputs/features rather than use all.

```python
with torch.enable_grad():
    reg = 1e-6
    l1_loss = torch.zeros(1)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + (reg * torch.sum(torch.abs(W)))
```

### Orthogonal Regularization

Improves gradient flow by keeping the matrix norm close to unitary.

```python
with torch.enable_grad():
    reg = 1e-6
    orth_loss = torch.zeros(1)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            param_flat = param.view(param.shape[0], -1)
            sym = torch.mm(param_flat, torch.t(param_flat))
            sym -= torch.eye(param_flat.shape[0])
            orth_loss = orth_loss + (reg * sym.abs().sum())
```

- [arXiv](https://arxiv.org/abs/1609.07093)

### Max Norm Constraint

If a hidden unit's weight vector's L2 norm `L` ever gets bigger than a certain max value `c`, multiply the weight vector by `c/L`. Enforce it immediately after each weight vector update or after every `X` gradient update.

This constraint is another form of regularization. While L2 penalizes high weights using the loss function, "max norm" acts directly on the weights. L2 exerts a constant pressure to move the weights near zero which could throw away useful information when the loss function doesn't provide incentive for the weights to remain far from zero. On the other hand, "max norm" never drives the weights to near zero. As long as the norm is less than the constraint value, the constraint has no effect.

```python
def max_norm(model, max_val=3, eps=1e-8):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))
```

- [Google+ Discussion](https://plus.google.com/+IanGoodfellow/posts/QUaCJfvDpni)

## Batch Normalization

[...]

## Dropout

[...]

## Optimization Misc.

- Learning Rate
- Batch Size
- Optimizer
- Generalization

- [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836)
- [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489)
- [Reddit Discussion](https://www.reddit.com/r/MachineLearning/comments/77dn96/r_171006451_understanding_generalization_and/dol2u23/)

## Correct Validation Strategies

[...]

- [Reddit Discussion](https://www.reddit.com/r/MachineLearning/comments/78789r/d_is_my_validation_method_good/)


## References

- Thanks to [Zijun Deng](https://github.com/zijundeng/pytorch-semantic-segmentation) for inspiring the code for the segmentation metrics.