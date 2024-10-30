import torch
from sklearn.metrics import r2_score
from torch import nn

ARCH = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

NUM_EPOCHS = 1000

SIZE_TRAIN_SET = 1000

SIZE_TEST_SET = 1000


def get_dataset(size):
    X = torch.rand((size, 2)) * 100
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = x1 + x2
    return X, y


class MyMachine(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = ARCH

    def forward(self, x):
        x = self.fc(x)
        return x


def train():
    model = MyMachine()
    model.load_state_dict(torch.load("model.h5"))
    model.train()
    X, y = get_dataset(SIZE_TRAIN_SET)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
    criterion = torch.nn.MSELoss(reduction='mean')

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        y_pred = model(X)
        y_pred = y_pred.reshape(1000)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch}, Loss:{loss.item()}')
    torch.save(model.state_dict(), 'model.h5')


def test():
    model = MyMachine()
    model.load_state_dict(torch.load("model.h5"))
    model.eval()
    X, y = get_dataset(SIZE_TEST_SET)

    with torch.no_grad():
        y_pred = model(X)
        print(r2_score(y, y_pred))


# train()
# test()

n1 = float(input())
n2 = float(input())

model = MyMachine()
model.load_state_dict(torch.load("model.h5"))
model.eval()

print(model(torch.Tensor([[n1, n2]])))
