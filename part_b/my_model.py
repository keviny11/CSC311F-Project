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
from part_a.item_response import irt
from part_a.item_response import evaluate as ir_eval

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
    def __init__(self, num_question, k=32):
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
    val_pred = evaluate(model, zero_train_data, valid_data, out=True)
    test_pred = evaluate(model, zero_train_data, test_data, out=True)
    return val_pred, test_pred


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
    # print("bagging accuracy", correct/len(predictions))
    return correct/len(predictions)

def bagging_auto (k,lr,lamb,num_epoch,num_auto):
    base_path = "../data"
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    val_pred_all = np.zeros(len(valid_data['user_id']))
    test_pred_all = np.zeros(len(test_data['user_id']))
    for i in range (num_auto):
        model = AutoEncoder(1774, k=k)
        train_matrix = load_train_sparse(base_path).toarray()
        zero_train_matrix = train_matrix.copy()
        # Fill in the missing entries to 0.
        zero_train_matrix[np.isnan(train_matrix)] = 0
        # Change to Float Tensor for PyTorch.
        zero_train_matrix = torch.FloatTensor(zero_train_matrix)
        train_matrix = torch.FloatTensor(train_matrix)
        val_pred, test_pred = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, test_data)
        val_pred_all += np.array(val_pred)
        test_pred_all += np.array(test_pred)
        print("autoencoder", i,"done")
    avg_val_pred = val_pred_all/num_auto
    avg_test_pred = test_pred_all / num_auto
    val_acc = eval_overall(avg_val_pred, valid_data)
    test_acc = eval_overall(avg_test_pred, test_data)
    print('valid acc of ensemble autoencoder is', val_acc)
    print('test acc of ensemble autoencoder is', test_acc)
    return val_acc, test_acc

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

######################## ensemble kmeans, item response and autoencoder
    ## item response
    train_data = load_train_csv("../data")
    ir_lr = 0.05
    iterations = 30
    t, b, t_acc, v_acc, t_neg, v_neg, t_neg_avg, v_neg_avg, prob_t, prob_v = irt(train_data, valid_data, ir_lr, iterations)
    prob_test = ir_eval(test_data, t, b)[1]
    print('item response done')
    # autoencoder
    pred_nn_val, pred_nn_test = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, test_data)
    print('autoencoder done')
    # k-means
    val_pred_kmeans, val_acc_kmeans = k_means_category(5, valid_data)
    test_pred_kmeans, test_acc_kmeans = k_means_category(5, test_data)
    print('k-means done')
    # ensemble
    bagged_val_pred = (0.8*np.array(prob_v)+0.2*np.array(pred_nn_val)+0.0*np.array(val_pred_kmeans))
    bag_val_acc = eval_overall(bagged_val_pred, valid_data)
    print('bagged validation accuracy is', bag_val_acc)
    bagged_test_pred = (0.8*np.array(prob_test)+0.2*np.array(pred_nn_test)+0.0*np.array(test_pred_kmeans))
    bag_test_acc = eval_overall(bagged_test_pred, test_data)
    print('bagged test accuracy is', bag_test_acc)

############### ensemble autoencoder
    # num_auto = 3
    # val_acc, test_acc = bagging_auto (k,lr,lamb,num_epoch,num_auto)
    # return val_acc, test_acc

if __name__ == "__main__":
    # val_acc, test_acc = main()
    main()
    # k= 100, inner = 64, epoch = 30, lr = 0.05, lamb = 0.001, val 69.24, test 69.7%
    # k= 256, inner = 128, epoch = 30, lr = 0.05, lamb = 0.001, val 69.00, test 69.06%
    # k= 64, inner = 32, epoch = 30, lr = 0.05, lamb = 0.001, val 69.26, test 70.22%
    # k = 128, inner = 64, 0.4, 0.6, val 69.37 test 69.71
    # k = 128, inner 64, 0.3, 0.7, val 69.47 test 69.48
    # k = 64, inner 32, 0.4, 0.6, val 69.12 test 69.68
    # k = 64, inner 32, 0.6, 0.4, val 69.11 test 70.11
    # k = 64, inner 32, 0.7, 0.3, val 69.53 test 69.94
    # k = 64, inner 32, 0.8, 0.2, val 69.46 test 69.77
    # k = 64, inner 32, 0.2, 0.8, val 69.39 test 70.16
    # k = 64, inner 32, 0.2, 0.8, val 69.30 test 69.83

    # 0.8 item response + 0.2 nn: val 70.54, test 70.90