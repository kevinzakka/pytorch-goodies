## Weight Initialization

* Xavier initialization (general-purpose)
* He et. al initialization (ReLU activation)
* orthogonal initialization
* Normal(0, 1/N) (SELU activation)

```python
# Xavier init
for m in model:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal(m.weight)

# He et. al init
for m in model:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal(m.weight)

# orthogonal init
for m in model:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal(m.weight)

# SELU init
for m in model:
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal(m.weight, 0, sqrt(1. / n))
    elif isinstance(m, nn.Linear):
        n = m.out_features
        nn.init.normal(m.weight, 0, sqrt(1. / n))
```

For `BatchNorm` we initialize the weights to 1 and the biases to 0.

```python
for m in model:
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)
```

## Weight Regularization

* L2 Regularization: add L2 norm weight penalty to loss function.
* L1 Regularization: add L1 norm weight penalty to loss function.
* Orthogonal Regularization: apply a weight penalty of `|W*W.T - I|` to loss function.
* Max Norm Constraint: clamp weight norm to less than a constant `W.norm(2) < c`.

#### L2 Regularization

[...]

#### L1 Regularization

[...]

#### Orthogonal Regularization

[...]

#### Max Norm Constraint

If a hidden unit's weight vector's L2 norm `L` ever gets bigger than a certain max value `c`, multiply the weight vector by `c/L`. Enforce it immediately after each weight vector update or after every `X` gradient update.

- https://plus.google.com/+IanGoodfellow/posts/QUaCJfvDpni

```python
# l2 reg
l2_loss = Variable(torch.FloatTensor(1), requires_grad=True)
for W in model.parameters():
    l2_loss = l2_loss + (0.5 * W.norm(2) ** 2)

# l1 reg
l1_loss = Variable(torch.FloatTensor(1), requires_grad=True)
for W in model.parameters():
    l1_loss = l1_loss + W.norm(1)

# orthogonal reg (might be incorrect)
orth_loss = Variable(torch.FloatTensor(1), requires_grad=True)
for W in model.parameters():
    W_reshaped = W.view(W.shape[0], -1)
    sym = torch.mm(W_reshaped, torch.t(W_reshaped))
    sym -= Variable(torch.eye(W_reshaped.shape[0]))
    orth_loss = orth_loss + sym.sum()

# max norm constraint
def max_norm(model, max_val=3, eps=1e-8):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True) ** 2
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))
```
