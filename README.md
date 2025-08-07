# SimpleGrad - Modular Autograd Implementation

A simple autograd implementation with a modular structure to avoid circular dependencies.

## Project Structure

```
simplegrad/
├── __init__.py              # Main package exports
├── Tensor/
│   ├── __init__.py          # Tensor package exports
│   └── Tensor.py            # Core Tensor class
├── Ops/
│   ├── __init__.py          # Operations package exports
│   ├── Function.py          # Base Function class
│   ├── arithmetic.py        # Arithmetic operations (Add, Sub, Mul, Div, Pow)
│   └── registration.py      # Operation registration utilities
└── testing.ipynb            # Jupyter notebook for testing
```

## Key Design Decisions

### 1. **Separation of Concerns**
- **Tensor**: Contains only the core `Tensor` class and its basic functionality
- **Ops**: Contains all operations and the base `Function` class
- **Registration**: Handles the registration of operations to the Tensor class

### 2. **Circular Dependency Avoidance**
- Used `TYPE_CHECKING` for type hints to avoid circular imports
- Lazy imports in the `Function.apply` method
- Registration happens at module import time with proper error handling

### 3. **Modular Operation Structure**
- Base `Function` class in `Ops/Function.py`
- Arithmetic operations grouped in `Ops/arithmetic.py`
- Easy to add new operation categories (e.g., `Ops/activation.py`, `Ops/loss.py`)

## Usage

```python
from Tensor import Tensor

# Create tensors
a = Tensor([5])
b = Tensor([2])
c = Tensor([2.3])

# Perform operations
res = a**2 * c / 4 * 2.4 / b

# Compute gradients
res.backward()

print("A grad:", a.grad)
print("B grad:", b.grad)
print("RES:", res)
```

## Adding New Operations

To add a new operation:

1. **Create the operation class** in the appropriate file (e.g., `Ops/arithmetic.py`):
```python
class NewOp(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y  # Your operation here
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output  # Your gradients here
```

2. **Register the operation** in `Tensor/Tensor.py`:
```python
def _register_operations():
    # ... existing registrations ...
    register("new_op", NewOp())
```

3. **Add magic method** (optional) in `Tensor/Tensor.py`:
```python
def __new_op__(self, other):
    return self.new_op(other)
```

## Benefits of This Structure

1. **Maintainability**: Each module has a single responsibility
2. **Extensibility**: Easy to add new operations without modifying core Tensor code
3. **Testability**: Operations can be tested independently
4. **No Circular Dependencies**: Clean import structure
5. **Type Safety**: Proper type hints with TYPE_CHECKING

## Testing

The original functionality is preserved and can be tested using the Jupyter notebook or by running:

```python
from Tensor import Tensor
# ... your test code ...
``` 