import numpy as np
from simplegrad import Tensor


def test_requires_grad_false():
    # Test that operations work correctly when requires_grad=False
    x = np.random.rand(3, 4)
    t_x = Tensor(x, requires_grad=False)

    # All operations should work without gradients
    res_exp = t_x.exp()
    res_relu = t_x.relu()
    res_sigmoid = t_x.sigmoid()
    res_log = t_x.log()
    res_neg = t_x.neg()
    res_softmax = t_x.softmax()

    # Should not have gradients
    assert t_x.grad is None
    assert res_exp.grad is None
    assert res_relu.grad is None
    assert res_sigmoid.grad is None
    assert res_log.grad is None
    assert res_neg.grad is None
    assert res_softmax.grad is None
