from simplegrad.Tensor import Tensor
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from simplegrad.optim import SGD
from simplegrad import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.l1 = nn.Linear(in_dim, 128)
        self.l2 = nn.Linear(128, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1.forward(x)
        x = x.relu()
        return self.l2.forward(x)


model = MLP(8 * 8, 10)
model.named_parameters()
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False
)
lr = 0.001
optim = SGD(params=model.parameters(), lr=lr)
for epoch in range(1):
    progress_bar = tqdm(zip(X_train, y_train), total=len(y_train), desc="loss: 0.0")
    for sample, target in progress_bar:
        sample = Tensor(sample, requires_grad=False)
        one_hot_target = np.zeros(10)
        one_hot_target[target] = 1
        target = Tensor(one_hot_target, requires_grad=False)

        logits = model.forward(sample.reshape((1, -1)))

        probs = logits.softmax()

        safe_probs = probs + 1e-8
        log_probs = safe_probs.log()
        loss_dot = target.dot(log_probs.reshape(-1))
        loss = -loss_dot

        optim.zero_grad()

        loss.backward()
        progress_bar.set_description_str(f"loss: {float(loss.item):.4f}")

        # perform step
        optim.step()


sample_idx = 12
print("TARGET: ", y_test[sample_idx])

x = Tensor(X_test[sample_idx])
res = model.forward(x)
print(np.argmax(res.item))
