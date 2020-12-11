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
from part_b.k_means_category import k_means_category
from part_b.cosine_similarity import cosine_similarity_students
from part_b.random_forest import random_forest


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
        self.g = nn.Linear(num_question, 64)
        self.f = nn.Linear(64, k)
        self.e = nn.Linear(k, 64)
        self.h = nn.Linear(64, num_question)

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


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, test_data):
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

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        # Store values for plotting
        losses.append(train_loss)
        accs.append(valid_acc)

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

    return evaluate(model, zero_train_data, test_data, out=True)

def evaluate(model, train_data, valid_data, out=False):
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

    predictions = []

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        probability = output[0][valid_data["question_id"][i]].item()
        if out:
            predictions.append(probability)

        guess = probability >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1

    if out: # output prediction instead of accuracy if out=True
        return predictions
    return correct / float(total)

def eval_overall(predictions, valid_data):
    correct = 0
    for i, t in enumerate(valid_data["is_correct"]):
        guess = predictions[i] >= 0.5
        if guess == t:
            correct += 1
    print(correct/len(predictions))

def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Set model hyperparameters.

    # k = 10: 67.60%
    # k = 50: 68.33%
    # k = 100: 68.50%
    # k = 200: 68.34%
    # k = 500: 67.33%
    k = 32

    # Set optimization hyperparameters.
    lr = 0.05  # options explored: 0.1, 0.01, 0.005
    num_epoch = 30

    # lamb = 0.001: 68.90%(valid), 67.88(test)
    # lamb = 0.01: 68.25%(valid)
    # lamb = 0.1: 68.36%(valid)
    # lamb = 1: 62.51%(valid)
    lamb = 0.001

    models = []

    model = AutoEncoder(1774, k=k)
    pred_nn = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, test_data)

    # Bagging
    bagged_pred = []

    # pred_CS = cosine_similarity_students(threshold=1.0, top=2)
    # for i, v in enumerate(pred_CS):
    #     if v is None:
    #         bagged_pred.append(pred_nn[i])
    #     else:
    #         bagged_pred.append(0.7*v+0.3*pred_nn[i])

    # pred_kmeans, acc = k_means_category(200, test_data)
    # for i, v in enumerate(pred_kmeans):
    #     if v != 0:
    #         bagged_pred.append(0.4*v+0.6*pred_nn[i])
    #     else:
    #         bagged_pred.append(pred_nn[i])

    # pred_RF = random_forest()
    # for i, v in enumerate(pred_RF):
    #     bagged_pred.append(0.5*pred_RF[i]+0.5*pred_nn[i])
    #
    # eval_overall(bagged_pred, test_data)


if __name__ == "__main__":
    main()
    # k= 100, inner = 64, epoch = 30, lr = 0.05, lamb = 0.001, val 69.24, test 69.7%
    # k= 256, inner = 128, epoch = 30, lr = 0.05, lamb = 0.001, val 69.00, test 69.06%
    # k= 64, inner = 32, epoch = 30, lr = 0.05, lamb = 0.001, val 69.26, test 70.22%