# pyright: strict
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from typing import List, Any, Tuple
from tiny_autograd import *


class AutodiffTest(unittest.TestCase):
    def assertGrad(
        self,
        result: Tuple[ValueType, Grads],
        output_expected: ValueType,
        **grads_expected: ValueType | List[Any],
    ):
        output, grads = result
        assert_almost_equal(output, output_expected, err_msg="function output mismatch")
        for name, grad in grads:
            grad_expected = grads_expected.get(name)
            if grad_expected is None:
                raise AssertionError(
                    f'argument "{name}" not provided, returned by grad:\n {repr(grad)}'
                )
            assert_almost_equal(
                grad,
                grad_expected,
                decimal=2,
                err_msg=f'argument "{name}" mismatch',
            )

    def test_sum_and_mean(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        sum = grad(lambda a: a.sum(axis=1).exp().sum())
        self.assertGrad(
            sum(a),
            3269420.8012656034,
            a=[
                [4.03428793e02, 4.03428793e02, 4.03428793e02],
                [3.26901737e06, 3.26901737e06, 3.26901737e06],
            ],
        )

        mean = grad(lambda a: a.mean(axis=1).pow(2).sum())
        self.assertGrad(
            mean(a),
            29,
            a=[[1.3333, 1.3333, 1.3333], [3.3333, 3.3333, 3.3333]],
        )

    def test_arithm(self):
        pass

    def test_matmul(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([7.0, 9.0, 11.0])
        c = np.array([7.0, 3.0])
        d = np.array([[1.5, 1.0], [2.5, 5.0], [7.0, 3.5]])
        f = grad(lambda a, b: (a @ b).pow(2).sum())

        # check one dimensional expansion
        self.assertGrad(
            f(a, b),
            22685.0,
            a=[[812.0, 1044.0, 1276.0], [1946.0, 2502.0, 3058.0]],
            b=[1228.0, 1622.0, 2016.0],
        )
        self.assertGrad(
            f(c, a),
            2723.0,
            a=[388.0, 910.0],
            b=[[266.0, 406.0, 546.0], [114.0, 174.0, 234.0]],
        )

        # two matrices
        self.assertGrad(
            f(a, d),
            7378.75,
            a=[[125.5, 352.5, 535.5], [281.5, 802.5, 1197.0]],
            b=[[539.0, 443.0], [715.0, 586.0], [891.0, 729.0]],
        )

        # batch dimension
        self.assertGrad(
            f(np.stack([a, a]), b),
            45370.0,
            a=[
                [[812.0, 1044.0, 1276.0], [1946.0, 2502.0, 3058.0]],
                [[812.0, 1044.0, 1276.0], [1946.0, 2502.0, 3058.0]],
            ],
            b=[2456.0, 3244.0, 4032.0],
        )
        self.assertGrad(
            f(np.stack([a, a]), np.stack([b, b, b], axis=1)),
            136110.0,
            a=[
                [[2436.0, 3132.0, 3828.0], [5838.0, 7506.0, 9174.0]],
                [[2436.0, 3132.0, 3828.0], [5838.0, 7506.0, 9174.0]],
            ],
            b=[
                [2456.0, 2456.0, 2456.0],
                [3244.0, 3244.0, 3244.0],
                [4032.0, 4032.0, 4032.0],
            ],
        )

    def test_softmax(self):
        x = np.array([[100.0, 90.0, 99.0], [80.0, 81.0, 78.0]])
        m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        sm = grad(lambda x: (x.softmax() * m[0]).sum())
        self.assertGrad(
            sm(x[0]),
            1.5378983,
            x=[-3.9322203e-01, 1.5336655e-05, 3.9320675e-01],
        )

        sm = grad(lambda x: (x.softmax(axis=0) * m).sum())
        self.assertGrad(
            sm(x),
            6.000370192186187,
            x=[
                [0.0, -3.7026405e-04, 0.0],
                [6.1834609e-09, 3.7013800e-04, 2.2747679e-09],
            ],
        )

        sm = grad(lambda x: (x.softmax(axis=1) * m).sum())
        self.assertGrad(
            sm(x),
            6.313520746520074,
            x=[
                [-3.9322203e-01, 1.5336655e-05, 3.9320675e-01],
                [-2.0127125e-01, 1.5827250e-01, 4.2998947e-02],
            ],
        )

    def test_linear_regression(self):
        np.random.seed(12345)
        xs = np.random.normal(size=(100,))
        noise = np.random.normal(scale=0.1, size=(100,))
        ys = xs * 3.0 - 1.0 + noise

        def model(theta: Var, x: Var) -> Var:
            w, b = theta
            return w * x + b

        def loss_fn(theta: Var, x: Var, y: Var) -> Var:
            prediction = model(theta, x)
            return ((prediction - y) ** 2).mean()

        loss_fn_grad = grad(loss_fn, argnum={0})

        def update(theta: ValueType, lr: float = 0.1) -> ValueType:
            _, grads = loss_fn_grad(theta, xs, ys)
            return theta - lr * grads.theta

        theta = np.array([1.0, 1.0])
        for _ in range(100):
            theta = update(theta)
        assert_almost_equal(theta, [2.99636699, -1.00343618])

    def test_grad_leaves(self):
        tape = Tape()
        x = tape.var(np.array([-1, 2, 0]))
        y = tape.var(np.array([0, 1, 2]))
        z = tape.var(1.0)
        o = ((x - y) ** (z * z * 2.0)).sum()
        grad = o.grad(leaves=[x])
        assert_almost_equal(grad.wrt(x), [-2, 2, -4])
        assert_almost_equal(grad.wrt(y), 0.0)  # not computed
        assert_almost_equal(grad.wrt(z), 0.0)  # not computed


if __name__ == "__main__":
    unittest.main()
