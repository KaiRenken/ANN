import torch
from sklearn.metrics import r2_score

from AdditionNetwork import AdditionNetwork
from data_loader import get_dataset

NUM_EPOCHS = 1000

SIZE_TRAIN_SET = 1000

SIZE_TEST_SET = 1000


def train():
    model = AdditionNetwork()
    model.load_state_dict(torch.load("model/model.h5"))
    X, y = get_dataset(SIZE_TRAIN_SET)

    model.train_network(X, y, NUM_EPOCHS)
    torch.save(model.state_dict(), 'model/model.h5')


def test():
    model = AdditionNetwork()
    model.load_state_dict(torch.load("model/model.h5"))
    X, y = get_dataset(SIZE_TEST_SET)

    print(r2_score(y, model.predict(X)))


train()
test()
