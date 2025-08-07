from simplegrad.Tensor import Tensor
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from simplegrad.optim import SGD, Adam
from simplegrad import nn
from simplegrad.nn import functional as F

np.random.seed(23)


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


model = MLP(8 * 8, 10)
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False
)
lr = 0.01
optim = Adam(params=model.parameters(), lr=lr)
batch_size = 32
num_batches = int(np.ceil(len(X_train) / batch_size))
for epoch in range(10):
    progress_bar = tqdm(range(num_batches), total=num_batches, desc="loss: 0.0")
    for batch_idx in progress_bar:
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(X_train))
        batch_samples = X_train[start:end]
        batch_targets = y_train[start:end]

        samples = Tensor(batch_samples, requires_grad=False)
        one_hot_targets = np.zeros((len(batch_targets), 10))
        one_hot_targets[np.arange(len(batch_targets)), batch_targets] = 1
        targets = Tensor(one_hot_targets, requires_grad=False)

        logits = model.forward(samples)

        probs = logits.softmax()

        safe_probs = probs + 1e-8
        log_probs = safe_probs.log()
        loss_dot = (targets * log_probs).sum(axis=-1)
        loss = -loss_dot
        loss = loss.mean()

        optim.zero_grad()

        loss.backward()
        progress_bar.set_description_str(f"loss: {float(loss.item):.4f}")

        optim.step()


sample_idx = 4
print("TARGET: ", y_test[sample_idx])

x = Tensor(X_test[sample_idx])
res = model.forward(x)
print(np.argmax(res.item))
