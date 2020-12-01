from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # log_lklihood = 0.
    prob = 0
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q])
        prob += np.log(sigmoid(x))
    log_lklihood = prob
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        dtheta = 1 - sigmoid(x)
        theta[u] = theta[u] + lr * dtheta
        # compute x with updated theta
        x = theta[u] - beta[q]
        dbeta = -1 + sigmoid(x)
        beta[q] = beta[q] + lr * dbeta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.random(max(data['user_id'])+1)
    beta = np.random.random(max(data['question_id'])+1)
    train_acc_lst = [] ### added for ploting training curve
    neg_lld_train = [] ### added for ploting training curve
    neg_lld_val = [] ### added for ploting validation curve
    val_acc_lst = []

    for i in range(iterations):
        neg_lld_v = neg_log_likelihood(val_data, theta=theta, beta=beta) ### added for ploting validation curve
        score_v, prob_v = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score_v)
        score_t, prob_t = evaluate(data=data, theta=theta, beta=beta) ### added for ploting training curve
        train_acc_lst.append(score_t) ### added for ploting training curve
        neg_lld_t = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_train.append(neg_lld_t)  ### added for ploting training curve
        neg_lld_val.append(neg_lld_v) ### added for ploting validation curve
        print("NLLK: {} \t Score: {}".format(neg_lld_v, score_v)) # print neg likelihood and accuracy for val set
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_lst, val_acc_lst, neg_lld_train, neg_lld_val, prob_t, prob_v


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    prob = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        prob.append(p_a)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"]), prob


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    import matplotlib.pyplot as plt
    # tune learning rate
    lr = [0.001,0.005,0.01,0.05,0.1]
#    lr = [0.05]
    for i in range (len(lr)):
        iterations = 10
        t,b,t_acc, v_acc, t_neg, v_neg, prob_t, prob_v = irt(train_data, val_data, lr[i], iterations)
        iter = np.linspace (0,iterations,iterations)
        plt.figure()
        # plt.plot(iter, t_acc)
        # plt.plot(iter, v_acc)
        plt.plot(iter, t_neg)
        plt.plot(iter, v_neg)
        plt.title('Negative log-likelihood')
        plt.ylabel('Negative log-likelihood')
        plt.xlabel('Number of Iteration')
        plt.legend(['Training', 'Validation'], loc='upper right')
        # plt.xticks(np.arange(1, 21, 1))
        plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    # Final accuracy with trained theta and beta
    final_val_acc = evaluate(val_data,t,b)
    final_test_acc = evaluate(test_data, t, b)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return t,b,t_acc, v_acc, t_neg, v_neg, prob_t, prob_v, final_val_acc, final_test_acc

if __name__ == "__main__":
    t,b,t_acc, v_acc, t_neg, v_neg, prob_t, prob_v, final_val_acc, final_test_acc  = main()
    import matplotlib.pyplot as plt

    plt.figure()
    # choose five questions
    for i in [10,20,30,40,50]:
        x = t - b[i]
        prob = sigmoid(x)
        plt.scatter(t, prob,s=10.)
    plt.title('Negative log-likelihood')
    plt.ylabel('Probability of correct response')
    plt.xlabel('Theta')
    plt.legend(['Question 1', 'Question 2', 'Question 3', 'Question 4', 'Question 5'], loc='upper right')
    plt.show()




