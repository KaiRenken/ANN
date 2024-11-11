import torch
from torch import nn


class AdditionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.reshape = 1000

    def forward(self, x):
        x = self.sequential(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            prediction = self(x)
        return prediction

    def train_network(self, X, y, epochs):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss(reduction='mean')
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self(X)
            y_pred = y_pred.reshape(self.reshape)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch}, Loss:{loss.item()}')
