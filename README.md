## simplegrad

Minimal CPU only autodiff and neural network library built on top of NumPy. It implements a small Tensor type with a dynamic computation graph and reverse‑mode automatic differentiation, plus a lightweight `nn` and `optim` layer for building and training simple (or even complex) models.

Despite its simplicity, the primitives here are enough to compose most building blocks needed for even complex architectures. I built this project to deepen my understanding of frameworks like PyTorch, and I hope it helps others grasp how PyTorch works by studying a minimal, NumPy‑only implementation of its core ideas (tensors, autograd, modules, optimizers).

Core library dependency: only NumPy.

### Folder structure

- **`simplegrad/Tensor/`**: `Tensor` implementation and autograd entry points
  - **`Tensor.py`**: tensor data container (`item` as `np.ndarray`), autograd graph, `backward()`
- **`simplegrad/Ops/`**: differentiable operations registered onto `Tensor`
  - **`Function.py`**: base class for all ops
  - **`UOps.py` / `BOps.py`**: unary/binary ops (e.g., `relu`, `exp`, `log`, `add`, `mul`, `dot`, `softmax`, `cross_entropy_loss`)
  - **`registration.py`**: attaches ops to the `Tensor` class
- **`simplegrad/nn/`**: tiny neural‑net utilities
  - **`Module.py`**: base module (`forward`, parameter discovery, save/load `state_dict`)
  - **`Parameter.py`**: marks trainable tensors (`requires_grad=True`)
  - **`Linear.py`**: affine layer (`x.dot(W) [+ b]`), Xavier init
  - **`Sequential.py`**: stacks modules/functions in a sequential manner
  - **`functional.py`**: functional wrappers (`relu`, `softmax`, `cross_entropy_loss`, ...)
  - **`MNISTLoader.py`**: optional example data loader (not needed for core library)
- **`simplegrad/optim/`**: optimizers
  - **`Optimizer.py`**: base optimizer class (`zero_grad`, param filtering)
  - **`SGD.py`**, **`Adam.py`**: implement the update rules
- **`simplegrad/tests/`**: unit tests for ops and tensor
- **`examples/`**: usage examples (uses additional dependencies like huggingface datasets)

### Minimal example (NumPy‑only)

```python
import numpy as np
from simplegrad import Tensor, nn, optim
from simplegrad.nn import functional as F

# Toy data: 2-class problem from a simple rule
rng = np.random.default_rng(0)
X_np = rng.normal(size=(128, 2)).astype(np.float32)
y_idx = (X_np[:, 0] + X_np[:, 1] > 0).astype(np.int64)
y_np = np.eye(2, dtype=np.float32)[y_idx]

X = Tensor(X_np)
y = Tensor(y_np)

model = nn.Sequential([
    nn.Linear(2, 16),
    F.relu,
    nn.Linear(16, 2),
])

opt = optim.Adam(model.parameters(), lr=1e-2)

for _ in range(200):
    opt.zero_grad()
    logits = model(X)
    loss = logits.cross_entropy_loss(y)
    loss.backward()
    opt.step()

print("final loss:", loss.item)
```
