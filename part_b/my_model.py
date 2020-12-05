from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import sys
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=64):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.f = nn.Linear(k, 32)
        self.e = nn.Linear(32, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        f_w_norm = torch.norm(self.f.weight, 2)
        e_w_norm = torch.norm(self.e.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm + e_w_norm + f_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        sigmoid = nn.Sigmoid()
        tmp = sigmoid(self.g(inputs))
        tmp = sigmoid(self.f(tmp))
        tmp = sigmoid(self.e(tmp))
        out = sigmoid(self.h(tmp))
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, test_data, boost_freq, out=True):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # For plotting
    losses, accs = [], []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + 0.5 * lamb * model.get_weight_norm()  # add reg term
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        if epoch % boost_freq == 0:
            failed = extract_failed(model, train_data, zero_train_data, int(train_data.shape[0] / 3))
            for user_id in failed:
                inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
                target = inputs.clone()

                optimizer.zero_grad()
                output = model(inputs)

                # Mask the target to only compute the gradient of valid entries.
                nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
                target[0][nan_mask] = output[0][nan_mask]

                loss = torch.sum((output - target) ** 2.) + 0.5 * lamb * model.get_weight_norm()  # add reg term
                loss.backward()

                train_loss += loss.item()
                optimizer.step()

        if out:
            valid_acc = evaluate(model, zero_train_data, valid_data)
            print("Epoch: {} \tTraining Cost: {:.6f}\t "
                  "Valid Acc: {}".format(epoch, train_loss, valid_acc))

            # Store values for plotting
            losses.append(train_loss)
            accs.append(valid_acc)

    if out:
        # plt.subplot(1, 2, 1)
        # plt.plot(np.array(range(num_epoch)), losses)
        # plt.title("Loss vs Epochs")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.subplot(1, 2, 2)
        # plt.plot(np.array(range(num_epoch)), accs)
        # plt.title("Accuracy vs Epochs")
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.show()

        print("Final Test Acc: {}".format(evaluate(model, zero_train_data, test_data)))
    return losses, accs
def extract_failed(model, train_data, zero_train_data, n):
    model.eval()

    score = [] # store the indices of those

    for j, v in enumerate(zero_train_data):
        inputs = Variable(v).unsqueeze(0)
        output = model(inputs)

        inputs_ = inputs.detach().numpy().tolist()[0]

        output_ = output.detach().numpy().tolist()[0]

        total, correct = 0, 0
        for i in range(len(inputs_)):
            if not np.isnan(train_data[j][i]):
                if (inputs_[i] >= 0.5) == output_[i]:
                    correct += 1
                total += 1
        score.append(correct/total**2)

    # score = np.array(score)
    # print(score[score < 1])

    return np.argsort(score)[:n]


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

def sample_matrix(train_data, zero_train_data):
    rand_mat_idx = np.random.randint(train_data.shape[0], size=train_data.shape[0])
    return train_data[rand_mat_idx, :], zero_train_data[rand_mat_idx, :]

def evaluate_bagging(models, train_data, valid_data):
    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        output = None
        inputs = Variable(train_data[u]).unsqueeze(0)
        for model in models:
            output = model(inputs).detach().numpy()[0] if output is None else output + model(inputs).detach().numpy()[0]
        output = output / len(models)

        guess = output[valid_data["question_id"][i]] >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)



def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Set model hyperparameters.

    # k = 10: 67.60%
    # k = 50: 68.33%
    # k = 100: 68.50%
    # k = 200: 68.34%
    # k = 500: 67.33%
    k = 64

    # Set optimization hyperparameters.
    lr = 0.05  # options explored: 0.1, 0.01, 0.005
    num_epoch = 30

    # lamb = 0.001: 68.90%(valid), 67.88(test)
    # lamb = 0.01: 68.25%(valid)
    # lamb = 0.1: 68.36%(valid)
    # lamb = 1: 62.51%(valid)
    lamb = 0.001

    models = []

    # for i in range(3):
    #     model = AutoEncoder(1774, k=k)
    #
    #     # train_matrix_, zero_train_matrix_ = sample_matrix(train_matrix, zero_train_matrix)
    #
    #     # train(model, lr, lamb, train_matrix_, zero_train_matrix_, valid_data, num_epoch, test_data, 5)
    #
    #     models.append(model)
    model = AutoEncoder(1774, k=k)
    losses, accs = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, test_data, 5)

    # print(evaluate_bagging(models, zero_train_matrix, valid_data))
    return losses, accs

if __name__ == "__main__":
    losses, accs = main()
    # k= 100, inner = 64, epoch = 30, lr = 0.05, lamb = 0.001, val 69.24, test 69.7%
    # k= 256, inner = 128, epoch = 30, lr = 0.05, lamb = 0.001, val 69.00, test 69.06%
    # k= 64, inner = 32, epoch = 30, lr = 0.05, lamb = 0.001, val 69.26, test 70.22%