# pyright: strict
from __future__ import annotations

import inspect
from functools import wraps
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    NewType,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Dict,
)

import numpy as np
from numpy.typing import NDArray

__all__ = ["ArrayType", "ValueType", "Var", "Grads", "grad", "Tape"]

ArrayType = NDArray[np.float32 | np.float64]
ValueType = ArrayType | float
ValueIndex = NewType("ValueIndex", int)
VJP = Callable[[ValueType], ValueType]  # Vector-Jacobian product
T = TypeVar("T")


# ------------------------------------------------------------------------------
# Gradient
# ------------------------------------------------------------------------------


class Var:
    """Tracing variable

    All the operations are computing result (forward pass), recording dependencies
    and vector-jacobian product functions to the tape. By calling `grad` internal
    reference to the Tape is used to calculate all the gradients with respect to
    this variable.
    """

    __slots__ = ["_tape", "_value", "_index"]

    def __init__(self, tape: Tape, value: ValueType, index: ValueIndex) -> None:
        self._tape = tape
        self._value = value
        self._index = index  # index of this variable in the list of nodes on the tape

    def grad(self, leaves: Optional[Iterable[Var]] = None) -> Grad:
        """Calculate gradients of this variable with respect to leaves
        (or all other the variables if leaves is None) that produced this variable.

        Note: This variable needs to be a scalar.
        """
        if np.size(self._value) != 1:
            raise ValueError("Gradient can only be calculated for scalar variables")

        # initialize grads
        grads: List[ValueType] = [0.0] * (self._index + 1)
        grads[-1] = 1.0  # gradient with respect to value itself is 1.0

        # find variables that are required to compute gradients for all the leaves
        skip: List[bool] = [False] * (self._index + 1)
        if leaves is not None:
            # skip all nodes that are not required to compute gradients for leaves
            def is_required(index: ValueIndex) -> bool:
                if index in leaves_indices:
                    return True
                required = False
                # do not use `any` as it only evaluates until first True
                for child, _ in self._tape.nodes[index]:
                    if is_required(child):
                        required = True
                if not required:
                    skip[index] = True
                return required

            leaves_indices = {var._index for var in leaves}
            is_required(self._index)

        # reverse pass
        for index in range(self._index, -1, -1):
            grad = grads[index]
            # accumulate gradients
            for dep_index, vjp in self._tape.nodes[index]:
                if skip[dep_index]:
                    continue
                grads[dep_index] += vjp(grad)

        return Grad(grads)

    def __lift(self, value: VarLike) -> Var:
        """Lift value to the variable"""
        if isinstance(value, Var):
            return value
        return self._tape.var(np.array(value, copy=False))

    def __unlift(self, value: Any) -> ValueType:
        if isinstance(value, Var):
            return value._value
        return value

    def __repr__(self) -> str:
        return f"Var({self._value})"

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: Any) -> NDArray[np.bool_] | bool:  # type: ignore
        return np.equal(self._value, self.__unlift(other))

    def __ne__(self, other: Any) -> NDArray[np.bool_] | bool:  # type: ignore
        return np.not_equal(self._value, self.__unlift(other))

    def __lt__(self, other: Any) -> NDArray[np.bool_] | bool:
        return np.less(self._value, self.__unlift(other))

    def __le__(self, other: Any) -> NDArray[np.bool_] | bool:
        return np.less_equal(self._value, self.__unlift(other))

    def __gt__(self, other: Any) -> NDArray[np.bool_] | bool:
        return np.greater(self._value, self.__unlift(other))

    def __ge__(self, other: Any) -> NDArray[np.bool_] | bool:
        return np.greater_equal(self._value, self.__unlift(other))

    @property
    def shape(self) -> Tuple[int, ...]:
        return np.shape(self._value)

    def __getitem__(self, selector: Any) -> Var:
        def vjp(g: ValueType) -> ValueType:
            out = np.zeros_like(self._value)
            out[selector] = g
            return out

        assert isinstance(self._value, np.ndarray)  # nosec
        selector = self.__unlift(selector)
        return self._tape.var(self._value[selector], (self._index, vjp))

    def __add__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        return self._tape.var(
            np.add(self._value, other._value),
            (self._index, unbroadcast_vjp(self._value, lambda g: g)),
            (other._index, unbroadcast_vjp(other._value, lambda g: g)),
        )

    def __radd__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        return other.__add__(self)

    def __sub__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        return self._tape.var(
            np.subtract(self._value, other._value),
            (self._index, unbroadcast_vjp(self._value, lambda g: g)),
            (other._index, unbroadcast_vjp(other._value, lambda g: -g)),
        )

    def __rsub__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        return other.__sub__(self)

    def __mul__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        return self._tape.var(
            np.multiply(self._value, other._value),
            (self._index, unbroadcast_vjp(self._value, lambda g: g * other._value)),
            (other._index, unbroadcast_vjp(other._value, lambda g: g * self._value)),
        )

    def __rmul__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        return other.__mul__(self)

    def __div__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        result = np.divide(self._value, other._value)
        return self._tape.var(
            result,
            (
                self._index,
                lambda g: unbroadcast(np.divide(g, other._value), self._value),
            ),
            (
                other._index,
                lambda g: unbroadcast(-result * g / other._value, other._value),
            ),
        )

    def __rdiv__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        return other.__div__(self)

    def __matmul__(self, other: VarLike) -> Var:
        """Matrix multiplication

        Given M = A x B, and the gradient ∇M:
            vjp_A = ∇M x Bᵀ
            vjp_B = Aᵀ x ∇M

        Matrix multiplication ein-sum notation "bij,bjk->bik":
            ∇M, M - bik
            A - bij
            B - bjk
            jvp_A - bik x bkj (swap jk->kj) => bij
            jvp_B - bji (swap ij->ji) x bik => bjk
        """
        a = self
        b = self.__lift(other)

        def vjp_a(grad: ValueType) -> ValueType:
            b_val = b._value
            if np.ndim(a._value) == 1:
                grad = np.expand_dims(grad, -2)
            if np.ndim(b_val) == 1:
                # we need expand dimensions since one dimensional vectors are
                #  always row in numpy, to get an outer product
                b_val = np.expand_dims(b_val, 0)
                grad = np.expand_dims(grad, -1)
            else:
                b_val = np.swapaxes(b_val, -1, -2)
            return unbroadcast(np.matmul(grad, b_val), a._value)

        def vjp_b(grad: ValueType) -> ValueType:
            a_val = a._value
            b_val = b._value
            if np.ndim(b._value) == 1:
                grad = np.expand_dims(grad, -1)
            if np.ndim(a_val) == 1:
                a_val = np.expand_dims(a_val, -1)
                grad = np.expand_dims(grad, 0)
            else:
                a_val = np.swapaxes(a_val, -1, -2)
            result = np.matmul(a_val, grad)
            if np.ndim(b_val) == 1:
                result = np.squeeze(result, np.ndim(grad) - 1)
            return unbroadcast(result, b._value)

        return self._tape.var(
            np.matmul(a._value, b._value),
            (a._index, vjp_a),
            (b._index, vjp_b),
        )

    def __rmatmul__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        return other.__matmul__(self)

    def __pow__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        result = self._value**other._value

        return self._tape.var(
            result,
            (
                self._index,
                unbroadcast_vjp(
                    self._value,
                    lambda g: other._value
                    * self._value ** np.where(other._value, other._value - 1, 1.0)
                    * g,
                ),
            ),
            (
                other._index,
                unbroadcast_vjp(
                    other._value,
                    lambda g: np.log(replace_zero(self._value, 1.0)) * result * g,
                ),
            ),
        )

    def pow(self, other: VarLike) -> Var:
        return self.__pow__(other)

    def __rpow__(self, other: VarLike) -> Var:
        other = self.__lift(other)
        return other.__pow__(self)

    def exp(self) -> Var:
        result = np.exp(self._value)
        return self._tape.var(result, (self._index, lambda g: result * g))

    def log(self) -> Var:
        return self._tape.var(
            np.log(self._value), (self._index, lambda g: g / self._value)
        )

    def sum(self, axis: Any = None) -> Var:
        axis = self.__unlift(axis)
        return self._tape.var(
            np.sum(self._value, axis=axis),  # type: ignore
            (self._index, lambda g: broadcast_to_match(g, self._value, axis)[0]),
        )

    def mean(self, axis: Any = None) -> Var:
        def vjp(grad: ValueType) -> ValueType:
            grad, repeats = broadcast_to_match(grad, self._value, axis)
            return grad / repeats

        axis = self.__unlift(axis)
        return self._tape.var(np.mean(self._value, axis=axis), (self._index, vjp))

    def clip(self, min: Any = None, max: Any = None) -> Var:
        min = self.__unlift(min)
        max = self.__unlift(max)
        result = np.clip(self._value, min, max)
        return self._tape.var(
            result,
            (
                self._index,
                lambda g: g * np.logical_and(result != min, result != max),
            ),
        )

    def softmax(self, axis: Any = None) -> Var:
        def vjp(grad: ValueType) -> ValueType:
            """vjp(g) = S * (g - ΣgS), where S - soft-max"""
            grad_sm = grad * softmax
            return grad_sm - softmax * np.sum(grad_sm, axis=axis, keepdims=True)  # type: ignore

        axis = self.__unlift(axis)
        x = np.exp(self._value - np.max(self._value, axis=axis, keepdims=True))  # type: ignore
        softmax = x / np.sum(x, axis=axis, keepdims=True)  # type: ignore
        return self._tape.var(softmax, (self._index, vjp))

    def relu(self) -> Var:
        return self.clip(0.0)


VarLike = Var | ValueType


class Grad:
    """List of all the gradients with respect to some variable"""

    __slots__ = ["grads"]

    def __init__(self, grads: List[ValueType]) -> None:
        self.grads = grads

    def wrt(self, var: Var) -> ValueType:
        """Fetch gradient with respect to the provided variable"""
        return self.grads[var._index]  # type: ignore


class Tape:
    """Record of the execution graph together with un-evaluated vector-jacobian products

    Each node corresponds to a variable (destination) and contains the list of pairs of
    back references to variables (sources) that produced this variable together with
    vector-jacobian product function, that given the gradient of the destination will give
    the gradient of the source that needs to be accumulated in (added to).
    """

    __slots__ = ["nodes"]

    nodes: List[List[Tuple[ValueIndex, VJP]]]

    def __init__(self) -> None:
        self.nodes = []

    def var(self, value: ValueType, *dep_vjp: Tuple[ValueIndex, VJP]) -> Var:
        """Record new variable

        Args:
            value: value of the variable
            dep_vjp: dependency index together with its vector-jacobian product function
        """
        self.nodes.append(list(dep_vjp))
        index = ValueIndex(len(self.nodes) - 1)
        return Var(self, value, index)


class Grads:
    """Dictionary like object that contains gradients"""

    __slots__ = ["__grads"]

    def __init__(self, grads: Dict[str | int, ValueType]) -> None:
        self.__grads = grads

    def __getitem__(self, name_or_position: int | str) -> ValueType:
        grad = self.__grads.get(name_or_position)
        if grad is None:
            raise KeyError(f"invalid gradient: {name_or_position}")
        return grad

    def __getattr__(self, name: str) -> ValueType:
        grad = self.__grads.get(name)
        if grad is None:
            raise AttributeError(f"invalid graient: {name}")
        return grad

    def __iter__(self) -> Iterator[Tuple[str, ValueType]]:
        for name, grad in self.__grads.items():
            if isinstance(name, int):
                continue
            yield name, grad

    def __repr__(self) -> str:
        return "\n".join(f"{name}: {grad}" for name, grad in self)


def grad(
    fn: Callable[..., Any],
    argnum: Optional[Set[str | int]] = None,
) -> Callable[..., Tuple[ValueType, Grads]]:
    """Gradient decorator

    Convert function to a new function which returns result as the first argument
    and all the gradients of the result with respect to the arguments
    """

    def grad(*args: Any, **kwargs: Any) -> Tuple[ValueType, Grads]:
        tape = Tape()
        var_args, var_kwargs = tree_map((args, kwargs), lambda value: tape.var(value))

        # find leaf variables of interest and its names
        var_named: Dict[str, Any] = {}
        spec = inspect.getfullargspec(fn).args
        for index, var in enumerate(var_args):
            if argnum is None or index in argnum:
                var_named[spec[index]] = var
        for name, var in var_kwargs.items():
            if argnum is None or name in argnum:
                var_named[name] = var

        # trace and compute gradients
        result = fn(*var_args, **var_kwargs)
        grad = result.grad(tree_iter(var_named))

        # fetch gradients
        grads = Grads(tree_map(var_named, lambda var: grad.wrt(var)))
        return (result._value, grads)

    return grad


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def broadcast_to_match(
    a: ValueType,
    b: ValueType,
    axis: int | Tuple[int] | None,
) -> Tuple[ArrayType, int]:
    """Broadcast `a` along `axis` to match `b`s shape"""
    b_shape = np.shape(b)
    repeats: int = np.prod(np.array(b_shape)[axis])  # type: ignore
    shape = np.array(b_shape)
    shape[list(axis) if isinstance(axis, tuple) else axis] = 1
    return np.broadcast_to(np.reshape(a, shape), b_shape), repeats


def unbroadcast(input: ValueType, target: ValueType) -> ValueType:
    """Sum values along broadcast axes in `input` to mach the shape of the `target`"""
    output = np.array(input, copy=False)
    # leading dimensions are summed first
    while np.ndim(output) > np.ndim(target):
        output = np.sum(output, axis=0)  # type: ignore
    # sum along dimensions equal to 1
    for axis, size in enumerate(np.shape(target)):
        if size == 1:
            output = np.sum(output, axis=axis, keepdims=True)  # type: ignore
    return output


def unbroadcast_vjp(target: ValueType, vjp_base: VJP) -> VJP:
    """Gradients need to be summed up along broadcasted axes

    When single variable effects multiple paths of the gradient, gradient with
    respect to this variable of the final scalar function needs to be summed due
    to the chain rule: df(x_i(u))/du = Σ df/dx_i * dx_i/du

    Or we can consider jacobian of the broadcast function which is going to include
    ones in the intersections of broadcast source (column) and destinations (rows).
    Then when we consider jacobian vector `gᵀ x J` that would mean we need to sum
    gradients along broadcasted axes.
    """

    @wraps(vjp_base)
    def vjp(grad: ValueType) -> ValueType:
        return unbroadcast(vjp_base(grad), target)

    return vjp


def replace_zero(x: ValueType, val: ValueType) -> ValueType:
    return np.where(x, x, val)


def tree_map(tree: Any, fn: Callable[[Any], Any]) -> Any:
    """Map tree like structure"""
    if isinstance(tree, dict):
        return {key: tree_map(value, fn) for key, value in tree.items()}  # type: ignore
    elif isinstance(tree, list):
        return [tree_map(value, fn) for value in tree]  # type: ignore
    elif isinstance(tree, tuple):
        return type(tree)(tree_map(value, fn) for value in tree)  # type: ignore
    return fn(tree)


def tree_iter(tree: Any) -> Iterator[Any]:
    """Iterator over all leaf elements of the tree"""
    if isinstance(tree, dict):
        for tree in tree.values():
            yield from tree_iter(tree)
    elif isinstance(tree, (list, tuple)):
        for tree in tree:
            yield from tree_iter(tree)
    yield tree
