import numpy as np
from simplegrad import Tensor
from .utils import get_numeric


atol = 1e-5


def _expect_inplace_modification_error(run_backward_fn):
    threw = False
    try:
        run_backward_fn()
    except AssertionError as e:
        assert "modified by an inplace operation" in str(e)
        threw = True
    assert threw, "Expected an inplace modification error, but none was raised"


def test_add_forward_backward_and_radd():
    x = np.random.randn(3, 4)
    y = np.random.randn(3, 4)

    t_x = Tensor(x, requires_grad=True)
    t_y = Tensor(y, requires_grad=True)

    # x + y
    out = t_x + t_y
    out.backward()

    # Numeric jacobian-vector product (with ones upstream) for elementwise op
    num_dx = get_numeric(lambda a, b: a + b, x, y, direction=0)
    num_dy = get_numeric(lambda a, b: a + b, x, y, direction=1)

    assert np.allclose(num_dx, t_x.grad, atol=atol)
    assert np.allclose(num_dy, t_y.grad, atol=atol)

    # radd: scalar + Tensor
    t_x.zero_grad()
    out2 = 2.5 + t_x
    out2.backward()
    assert np.allclose(t_x.grad, np.ones_like(x), atol=atol)


def test_sub_forward_backward_and_rsub():
    x = np.random.randn(5, 2)
    y = np.random.randn(5, 2)

    t_x = Tensor(x, requires_grad=True)
    t_y = Tensor(y, requires_grad=True)

    # x - y
    out = t_x - t_y
    out.backward()

    num_dx = get_numeric(lambda a, b: a - b, x, y, direction=0)
    num_dy = get_numeric(lambda a, b: a - b, x, y, direction=1)

    assert np.allclose(num_dx, t_x.grad, atol=atol)
    assert np.allclose(num_dy, t_y.grad, atol=atol)

    # rsub: scalar - Tensor
    t_x.zero_grad()
    out2 = 3.0 - t_x
    out2.backward()
    assert np.allclose(t_x.grad, -np.ones_like(x), atol=atol)


def test_mul_forward_backward_and_rmul_and_inplace_error():
    x = np.random.randn(4, 3)
    y = np.random.randn(4, 3)

    t_x = Tensor(x, requires_grad=True)
    t_y = Tensor(y, requires_grad=True)

    out = t_x * t_y
    out.backward()

    num_dx = get_numeric(lambda a, b: a * b, x, y, direction=0)
    num_dy = get_numeric(lambda a, b: a * b, x, y, direction=1)

    assert np.allclose(num_dx, t_x.grad, atol=atol)
    assert np.allclose(num_dy, t_y.grad, atol=atol)

    # rmul: scalar * Tensor
    t_x.zero_grad()
    out2 = -1.25 * t_x
    out2.backward()
    assert np.allclose(t_x.grad, -1.25 * np.ones_like(x), atol=atol)

    # Inplace modification between forward and backward should raise
    t_x2 = Tensor(np.random.randn(2, 2), requires_grad=True)
    t_y2 = Tensor(np.random.randn(2, 2), requires_grad=True)
    out_err = t_x2 * t_y2
    # mutate operand used by backward
    t_y2[0, 0] = 123.0
    _expect_inplace_modification_error(lambda: out_err.sum().backward())


def test_div_forward_backward_and_rtruediv_and_inplace_error():
    # keep denominators away from zero
    x = np.random.randn(3, 3)
    y = np.random.rand(3, 3) + 0.5

    t_x = Tensor(x, requires_grad=True)
    t_y = Tensor(y, requires_grad=True)

    out = t_x / t_y
    out.backward()

    num_dx = get_numeric(lambda a, b: a / (b + 1e-8), x, y, direction=0)
    num_dy = get_numeric(lambda a, b: a / (b + 1e-8), x, y, direction=1)

    assert np.allclose(num_dx, t_x.grad, atol=atol)
    assert np.allclose(num_dy, t_y.grad, atol=atol)

    # rtruediv: scalar / Tensor
    t_x.zero_grad()
    positive = np.abs(x) + 0.5
    t_pos = Tensor(positive, requires_grad=True)
    out2 = 5.0 / t_pos
    out2.backward()
    expected = -5.0 / (positive**2)
    assert np.allclose(t_pos.grad, expected, atol=atol)

    # Inplace modification should raise
    t_a = Tensor(np.random.rand(2, 2) + 0.5, requires_grad=True)
    t_b = Tensor(np.random.rand(2, 2) + 0.5, requires_grad=True)
    out_err = t_a / t_b
    t_a[0, 1] = 0.0
    _expect_inplace_modification_error(lambda: out_err.sum().backward())


def test_pow_forward_backward_and_rpow_and_inplace_error():
    # base strictly positive for exponent gradient
    base = np.random.rand(2, 3) + 0.5
    exp_scalar = 1.7

    t_base = Tensor(base, requires_grad=True)
    t_exp = Tensor(np.array([exp_scalar]), requires_grad=True)

    out = t_base**t_exp
    out.backward()

    num_dbase = get_numeric(
        lambda a, b: np.power(a, b), base, np.array([exp_scalar]), direction=0
    )
    # exponent is treated as scalar; use direction=1
    num_dexp_full = get_numeric(
        lambda a, b: np.power(a, b), base, np.array([exp_scalar]), direction=1
    )

    assert np.allclose(num_dbase, t_base.grad, atol=atol)
    # Grad wrt scalar exponent is sum of elementwise contributions
    assert np.allclose(
        np.array([num_dexp_full.sum()], dtype=t_exp.grad.dtype), t_exp.grad, atol=atol
    )

    # rpow: scalar ** Tensor
    c = 3.2
    t_e = Tensor(np.random.rand(2, 2), requires_grad=True)
    out2 = c**t_e
    out2.backward()
    expected = (c**t_e.item) * np.log(c)
    assert np.allclose(t_e.grad, expected, atol=atol)

    # Inplace modification should raise
    t_b2 = Tensor(np.random.rand(2, 2) + 0.5, requires_grad=True)
    t_e2 = Tensor(np.array([1.3]), requires_grad=True)
    out_err = t_b2**t_e2
    t_b2[0, 0] = 7.0
    _expect_inplace_modification_error(lambda: out_err.sum().backward())


def test_broadcasting_grads_for_scalar_rhs_and_lhs():
    x = np.random.randn(2, 3)
    t_x = Tensor(x, requires_grad=True)

    # Tensor scalar on RHS
    t_s = Tensor(np.array([2.0]), requires_grad=True)
    out = t_x + t_s
    out.backward()
    # grad wrt scalar is sum of upstream ones
    assert np.allclose(t_s.grad, np.array([x.size], dtype=t_s.grad.dtype), atol=atol)

    # Scalar on LHS (radd) with Tensor scalar
    t_x.zero_grad()
    t_s2 = Tensor(np.array([1.5]), requires_grad=True)
    out2 = t_s2 + t_x
    out2.backward()
    assert np.allclose(t_s2.grad, np.array([x.size], dtype=t_s2.grad.dtype), atol=atol)


def test_dot_forward_backward_various_shapes_and_inplace_error():
    # vector · vector -> scalar
    x = np.random.randn(5)
    y = np.random.randn(5)
    t_x = Tensor(x, requires_grad=True)
    t_y = Tensor(y, requires_grad=True)
    out = t_x.dot(t_y)
    out.backward()
    assert t_x.grad.shape == x.shape and t_y.grad.shape == y.shape
    assert np.allclose(t_x.grad, y, atol=atol)
    assert np.allclose(t_y.grad, x, atol=atol)

    # matrix · vector -> vector
    A = np.random.randn(4, 3)
    v = np.random.randn(3)
    t_A = Tensor(A, requires_grad=True)
    t_v = Tensor(v, requires_grad=True)
    out2 = t_A.dot(t_v)
    out2.backward()
    # upstream ones of shape (4,) -> grad_A = outer(ones, v), grad_v = A^T @ ones
    assert np.allclose(t_A.grad, np.outer(np.ones(4), v), atol=atol)
    assert np.allclose(t_v.grad, A.T @ np.ones(4), atol=atol)

    # vector · matrix -> vector
    u = np.random.randn(3)
    B = np.random.randn(3, 2)
    t_u = Tensor(u, requires_grad=True)
    t_B = Tensor(B, requires_grad=True)
    out3 = t_u.dot(t_B)
    out3.backward()
    # upstream ones of shape (2,) -> grad_u = ones @ B^T, grad_B = outer(u, ones)
    assert np.allclose(t_u.grad, np.ones(2) @ B.T, atol=atol)
    assert np.allclose(t_B.grad, np.outer(u, np.ones(2)), atol=atol)

    # matrix · matrix -> matrix
    M = np.random.randn(2, 3)
    N = np.random.randn(3, 4)
    t_M = Tensor(M, requires_grad=True)
    t_N = Tensor(N, requires_grad=True)
    out4 = t_M.dot(t_N)
    out4.backward()
    # upstream ones of shape (2,4) -> grad_M = ones @ N^T, grad_N = M^T @ ones
    ones = np.ones((2, 4))
    assert np.allclose(t_M.grad, ones @ N.T, atol=atol)
    assert np.allclose(t_N.grad, M.T @ ones, atol=atol)

    # Inplace modification should raise
    t_P = Tensor(np.random.randn(2, 2), requires_grad=True)
    t_Q = Tensor(np.random.randn(2, 2), requires_grad=True)
    out_err = t_P.dot(t_Q)
    t_Q[0, 1] = -3.14
    _expect_inplace_modification_error(lambda: out_err.sum().backward())


def test_requires_grad_masking_of_parents():
    x = np.random.randn(3, 3)
    y = np.random.randn(3, 3)

    t_x = Tensor(x, requires_grad=True)
    t_y = Tensor(y, requires_grad=False)

    out = t_x * t_y
    out.backward()
    # Only x should accumulate grad, y remains None
    assert t_x.grad is not None
    assert t_y.grad is None
