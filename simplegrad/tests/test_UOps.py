from .utils import get_numeric
import numpy as np
from simplegrad import Tensor
from simplegrad.Ops import Sigmoid, Relu, Softmax


atol = 1e-5


def test_exp():
    x = np.random.rand(4, 2)

    t_x = Tensor(x, requires_grad=True)
    res = t_x.exp()
    res.backward()

    numeric_gt = get_numeric(np.exp, x)

    assert np.allclose(numeric_gt, t_x.grad, atol=atol)


def test_sum():
    # Test sum over all elements
    x = np.random.rand(3, 4)
    print(np.sum(x[0]))
    t_x = Tensor(x, requires_grad=True)
    res = t_x.sum()
    res.backward()
    # necessary since we broadcast the shape during gradient calculation
    accumulated_grad = np.sum(t_x.grad)

    numeric_gt = get_numeric(np.sum, x)

    assert np.allclose(numeric_gt, accumulated_grad, atol=atol)

    # Test sum over specific axis
    t_x.zero_grad()
    res = t_x.sum(axis=0)
    res.backward()
    accumulated_grad = np.sum(t_x.grad, axis=0)
    numeric_gt = get_numeric(lambda x: np.sum(x, axis=0), x)
    assert np.allclose(numeric_gt, accumulated_grad, atol=atol)

    # Test sum over multiple axes
    t_x.zero_grad()
    res = t_x.sum(axis=(0, 1))
    res.backward()
    accumulated_grad = np.sum(t_x.grad, axis=(0, 1))
    numeric_gt = get_numeric(lambda x: np.sum(x, axis=(0, 1)), x)
    assert np.allclose(numeric_gt, accumulated_grad, atol=atol)


def test_reshape():
    x = np.random.rand(2, 3, 4)
    t_x = Tensor(x, requires_grad=True)

    # Test reshape to different shape
    res = t_x.reshape((2, 12))
    res.backward()

    # Reshape doesn't change the gradient computation, just the shape
    assert np.allclose(t_x.grad, np.ones_like(x), atol=atol)

    # Test reshape back to original shape
    t_x.zero_grad()
    res = t_x.reshape((2, 3, 4))
    res.backward()
    assert np.allclose(t_x.grad, np.ones_like(x), atol=atol)


def test_relu():
    x = np.random.randn(4, 3)  # Use randn to get negative values
    t_x = Tensor(x, requires_grad=True)
    res = t_x.relu()
    res.backward()

    numeric_gt = get_numeric(Relu._forward, x)
    assert np.allclose(numeric_gt, t_x.grad, atol=atol)

    # Test with all negative values
    x_neg = np.random.rand(4, 3) * -1
    t_x_neg = Tensor(x_neg, requires_grad=True)
    res_neg = t_x_neg.relu()
    res_neg.backward()

    # Gradient should be zero for negative inputs
    assert np.allclose(t_x_neg.grad, np.zeros_like(x_neg), atol=atol)


def test_sigmoid():
    x = np.random.randn(4, 3)
    t_x = Tensor(x, requires_grad=True)
    res = t_x.sigmoid()
    res.backward()

    numeric_gt = get_numeric(Sigmoid._forward, x)
    assert np.allclose(numeric_gt, t_x.grad, atol=atol)

    # Test with large positive values (should be close to 1)
    x_large = np.random.rand(4, 3) * 10
    t_x_large = Tensor(x_large, requires_grad=True)
    res_large = t_x_large.sigmoid()
    res_large.backward()

    numeric_gt_large = get_numeric(Sigmoid._forward, x_large)
    assert np.allclose(numeric_gt_large, t_x_large.grad, atol=atol)


def test_log():
    x = np.random.rand(4, 3) + 0.1  # Ensure positive values
    t_x = Tensor(x, requires_grad=True)
    res = t_x.log()
    res.backward()

    numeric_gt = get_numeric(np.log, x)
    assert np.allclose(numeric_gt, t_x.grad, atol=atol)

    # Test with values close to 1
    x_close_to_one = np.random.rand(4, 3) * 0.1 + 0.9
    t_x_close = Tensor(x_close_to_one, requires_grad=True)
    res_close = t_x_close.log()
    res_close.backward()

    numeric_gt_close = get_numeric(np.log, x_close_to_one)
    assert np.allclose(numeric_gt_close, t_x_close.grad, atol=atol)


def test_neg():
    x = np.random.randn(4, 3)
    t_x = Tensor(x, requires_grad=True)
    res = t_x.neg()
    res.backward()

    numeric_gt = get_numeric(lambda x: -x, x)
    assert np.allclose(numeric_gt, t_x.grad, atol=atol)

    # Test with all positive values
    x_pos = np.random.rand(4, 3)
    t_x_pos = Tensor(x_pos, requires_grad=True)
    res_pos = t_x_pos.neg()
    res_pos.backward()

    numeric_gt_pos = get_numeric(lambda x: -x, x_pos)
    assert np.allclose(numeric_gt_pos, t_x_pos.grad, atol=atol)


def test_softmax():
    # Test softmax with default parameters
    x = np.random.randn(3, 4)
    t_x = Tensor(x, requires_grad=True)
    res = t_x.softmax()
    res.backward()

    # For softmax, we need to test the gradient through the entire computation
    # Since it's a complex function, we'll test that the gradients are reasonable
    assert t_x.grad is not None
    assert t_x.grad.shape == x.shape

    # Numeric gradient check for default parameters
    numeric_gt = get_numeric(lambda x_: Softmax._forward(x_, axis=1, temperature=1), x)
    assert np.allclose(numeric_gt, t_x.grad, atol=atol)

    # Test softmax with custom axis
    t_x.zero_grad()
    res = t_x.softmax(axis=0)
    res.backward()
    assert t_x.grad is not None
    assert t_x.grad.shape == x.shape

    # Numeric gradient check for custom axis
    numeric_gt_axis0 = get_numeric(
        lambda x_: Softmax._forward(x_, axis=0, temperature=1), x
    )
    assert np.allclose(numeric_gt_axis0, t_x.grad, atol=atol)

    # Test softmax with custom temperature
    t_x.zero_grad()
    res = t_x.softmax(temperature=2.0)
    res.backward()
    assert t_x.grad is not None
    assert t_x.grad.shape == x.shape

    # Numeric gradient check for custom temperature
    numeric_gt_temp = get_numeric(
        lambda x_: Softmax._forward(x_, axis=1, temperature=2.0), x
    )
    assert np.allclose(numeric_gt_temp, t_x.grad, atol=atol)

    # Test softmax with both custom axis and temperature
    t_x.zero_grad()
    res = t_x.softmax(axis=1, temperature=0.5)
    res.backward()
    assert t_x.grad is not None
    assert t_x.grad.shape == x.shape

    # Numeric gradient check for both custom axis and temperature
    numeric_gt_axis1_temp = get_numeric(
        lambda x_: Softmax._forward(x_, axis=1, temperature=0.5), x
    )
    assert np.allclose(numeric_gt_axis1_temp, t_x.grad, atol=atol)


def test_softmax_numerical_stability():
    # Test softmax with large values (should be numerically stable)
    x = np.random.randn(2, 3) * 100
    t_x = Tensor(x, requires_grad=True)
    res = t_x.softmax()
    res.backward()

    # Should not have NaN or inf values
    assert not np.any(np.isnan(t_x.grad))
    assert not np.any(np.isinf(t_x.grad))

    # Test softmax with very small values
    x_small = np.random.randn(2, 3) * 0.01
    t_x_small = Tensor(x_small, requires_grad=True)
    res_small = t_x_small.softmax()
    res_small.backward()

    assert not np.any(np.isnan(t_x_small.grad))
    assert not np.any(np.isinf(t_x_small.grad))


def test_edge_cases():
    # Test with scalar tensors
    x_scalar = np.array([5.0])
    t_x_scalar = Tensor(x_scalar, requires_grad=True)
    res_scalar = t_x_scalar.exp()
    res_scalar.backward()

    numeric_gt_scalar = get_numeric(np.exp, x_scalar)
    assert np.allclose(numeric_gt_scalar, t_x_scalar.grad, atol=atol)

    # Test with 1D tensors
    x_1d = np.random.rand(5)
    t_x_1d = Tensor(x_1d, requires_grad=True)
    res_1d = t_x_1d.relu()
    res_1d.backward()

    numeric_gt_1d = get_numeric(Relu._forward, x_1d)
    assert np.allclose(numeric_gt_1d, t_x_1d.grad, atol=atol)

    # Test with 3D tensors
    x_3d = np.random.rand(2, 3, 4)
    t_x_3d = Tensor(x_3d, requires_grad=True)
    res_3d = t_x_3d.sigmoid()
    res_3d.backward()

    numeric_gt_3d = get_numeric(Sigmoid._forward, x_3d)
    assert np.allclose(numeric_gt_3d, t_x_3d.grad, atol=atol)


