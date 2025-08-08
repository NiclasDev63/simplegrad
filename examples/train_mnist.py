import os, sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simplegrad import nn
from simplegrad import Tensor
from simplegrad.nn import functional as F
from simplegrad import optim
from tqdm import tqdm, trange
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            [
                nn.Linear(in_dim, 128, use_bias=False),
                F.relu,
                nn.Linear(128, out_dim, use_bias=False),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_epoch(
    mnist_train_loader: nn.MNISTLoader, model: nn.Module, optim: optim.Optimizer
):
    losses = []
    progress_bar = tqdm(mnist_train_loader, desc="Train")
    for images, labels in progress_bar:
        optim.zero_grad()

        logits: Tensor = model(images)

        loss = logits.cross_entropy_loss(labels)
        loss.backward()
        optim.step()

        progress_bar.set_description(f"Train loss: {loss.item}")
        losses.append(loss.item)

    avg_loss = np.sum(losses) / len(losses)
    tqdm.write(f"Total train loss: {avg_loss:.6f}\n")


def eval(mnist_eval_loader: nn.MNISTLoader, model: nn.Module):
    losses = []
    correct = 0
    total = 0
    progress_bar = tqdm(mnist_eval_loader, desc="Eval")
    for images, labels in progress_bar:
        logits: Tensor = model(images)

        loss = logits.cross_entropy_loss(labels)
        losses.append(loss.item)

        # Calculate accuracy
        preds = np.argmax(logits.item, axis=-1)
        targets = np.argmax(labels.item, axis=-1)
        correct += np.sum(preds == targets)
        total += preds.shape[0]

        progress_bar.set_description(f"Eval loss: {loss.item}")

    avg_loss = np.sum(losses) / len(losses)
    accuracy = correct / total if total > 0 else 0.0
    tqdm.write(f"Total eval loss: {avg_loss:.6f}")
    tqdm.write("Eval accuracy: {:.2f}%\n".format(accuracy * 100))


def train(
    mnist_train_loader: nn.MNISTLoader,
    mnist_eval_loader: nn.MNISTLoader,
    model: nn.Module,
    optim: optim.Optimizer,
    epochs: int = 10,
    checkpoint_path: Optional[str] = None,
):
    for epoch in trange(epochs, desc="Epochs"):
        train_epoch(mnist_train_loader, model, optim)
        eval(mnist_eval_loader, model)

    # store model state (storing other states like optimizer state is currently not supported)
    if checkpoint_path:
        model.save_state_dict(path=checkpoint_path)


if __name__ == "__main__":
    BATCH_SIZE = 2048
    EPOCHS = 10

    mnist_train_loader = nn.MNISTLoader(
        batch_size=BATCH_SIZE, shuffle=True, split="train"
    )
    mnist_eval_loader = nn.MNISTLoader(batch_size=BATCH_SIZE, split="test")

    model = MLP(
        in_dim=mnist_train_loader.images_size_flattened,
        out_dim=mnist_eval_loader.num_classes,
    )

    optim = optim.Adam(model.parameters())

    ckpt_path = "./checkpoints/model_new.pickle"
    train(
        mnist_train_loader=mnist_train_loader,
        mnist_eval_loader=mnist_eval_loader,
        model=model,
        optim=optim,
        epochs=EPOCHS,
        checkpoint_path=ckpt_path,
    )
