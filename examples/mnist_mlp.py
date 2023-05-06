from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import time
from functools import partial
from typing import Any, Dict, Iterator, NamedTuple, Tuple

import numpy as np
import datasets

from tiny_autograd import *

INPUT_SIZE: int = 28 * 28
CLASSES: int = 10
CLASSES_MAP: ArrayType = np.eye(CLASSES)


def mnist_map(item: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        image=[
            np.array(img, dtype=np.float32).reshape(INPUT_SIZE) / 255.0
            for img in item["image"]
        ],
        label=[CLASSES_MAP[label] for label in item["label"]],
    )


def mnist_load() -> Tuple[ArrayType, ArrayType, ArrayType, ArrayType]:
    ds = datasets.load_dataset("mnist")
    train = ds["train"].map(mnist_map, batched=True)  # type: ignore
    test = ds["test"].map(mnist_map, batched=True)  # type: ignore
    return (
        np.stack(train["image"]).squeeze(-1),
        np.stack(test["image"]).squeeze(-1),
        np.stack(train["label"]),
        np.stack(test["label"]),
    )  # type: ignore


def batched(x: Any, y: Any, batch_size: int) -> Iterator[Tuple[Any, Any]]:
    count = x.shape[0] // batch_size
    for index in np.random.permutation(count):
        start = index * batch_size
        end = (index + 1) * batch_size
        yield x[start:end], y[start:end]


class Linear(NamedTuple):
    weights: ArrayType
    bias: ArrayType

    @staticmethod
    def init(input: int, output: int) -> Linear:
        xavier_sigma = np.sqrt(2.0 / (input + output))
        weights = np.random.randn(input, output) * xavier_sigma
        bias = np.zeros(output)
        return Linear(weights, bias)

    def __call__(self, input: Var) -> Var:
        return input @ self.weights + self.bias


class Model(NamedTuple):
    hidden: Linear
    output: Linear

    @staticmethod
    def init(hidden: int) -> Model:
        return Model(
            Linear.init(INPUT_SIZE, hidden),
            Linear.init(hidden, CLASSES),
        )

    def __call__(self, x: Var) -> Var:
        x = self.hidden(x).relu()
        return self.output(x).softmax(axis=1)


def cross_entropy(predicted: Var, expected: Var) -> Var:
    return (-predicted.log() * expected).sum()


def accuracy(y_hat: ArrayType, y: ArrayType) -> float:
    return np.mean(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))


@partial(grad, argnums={0}, has_aux=True)
def evaluate(model: Model, x: Var, y: Var) -> Tuple[Var, Var]:
    y_hat = model(x)
    loss = cross_entropy(y_hat, y)
    return loss, y_hat


def update(
    model: Model, x: ArrayType, y: ArrayType, lr: float
) -> Tuple[Model, float, float]:
    (loss, y_hat), grads = evaluate(model, x, y)
    model_updated = tree_map(lambda model, grads: model - lr * grads, model, grads[0])
    return model_updated, loss, accuracy(y_hat, y)


def main():
    print("Loading MNIST ...")
    X_train, X_test, y_train, y_test = mnist_load()
    print(f"train size: {X_train.shape[0]}")
    print(f"test size:  {X_test.shape[0]}")

    # hyper parameters
    hidden = 128
    batch_size = 32
    epoch_count = 10
    lr = 0.01

    model = Model.init(hidden)
    for epoch in range(epoch_count):
        print(f"\x1b[32mepoch: {epoch}\x1b[m")
        epoch_start = time.time()

        # train
        loss_epoch = 0.0
        accuracy_epoch = []
        for x_batch, y_batch in batched(X_train, y_train, batch_size):
            model, loss, acc = update(model, x_batch, y_batch, lr)
            loss_epoch += loss
            accuracy_epoch.append(acc)
        epoch_end = time.time()
        print(f"time:  {epoch_end - epoch_start:.3f}")
        print(f"train: accuracy={np.mean(accuracy_epoch):.6f}\tloss={loss_epoch:4f}")

        # evaluation
        y_test_hat = evaluate(model, X_test, y_test)[0][1]
        print(f"valid: accuracy={accuracy(y_test_hat, y_test):.6f}")


if __name__ == "__main__":
    np.random.seed(12345)
    main()
