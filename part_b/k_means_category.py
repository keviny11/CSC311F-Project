import numpy as np
import pandas as pd
from utils import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys
import matplotlib.pyplot as plt

def gen_probability(clster):

    train_m = load_train_sparse("../data")
    q_meta_cat = q_meta_k_means(clster)

    predictions = []
    for r in train_m.toarray():

        correct = [0]*clster
        total = [0]*clster
        for i, entry in enumerate(r):
            if np.isnan(entry):
                continue
            if entry:
                correct[q_meta_cat[i]] += 1
            total[q_meta_cat[i]] += 1

        guess = np.random.uniform()
        p = [correct[i]/total[i] if total[i] != 0 else guess for i in range(len(total))]

        probability = []
        for i, _ in enumerate(r):
            probability.append(p[q_meta_cat[i]])
        predictions.append(probability)

    return predictions

def q_meta_k_means(clstr):
    q_meta_data = pd.read_csv("../data/question_meta.csv")
    q_meta_data = q_meta_data.sort_values("question_id")

    q_meta = np.zeros((q_meta_data.shape[0], 388))
    for index, row in q_meta_data.iterrows():
        for sid in eval(row["subject_id"]):
            q_meta[int(index), int(sid)] = 1

    pca = PCA(2)
    projected = pca.fit_transform(q_meta)
    kmeans = KMeans(n_clusters=clstr, random_state=0).fit(projected)
    labels = kmeans.labels_

    plt.scatter(projected[:, 0], projected[:, 1],
                c=labels, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', clstr))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title("PCA and K-means analysis on question subject categorization")
    # plt.show()
    return labels

def k_means_category(k, test):
    predictions = gen_probability(k)
    predictions_ = []
    total, correct = 0, 0
    for i, u in enumerate(test["user_id"]):
        if predictions[u][test["question_id"][i]] is not None:
            predictions_.append(predictions[u][test["question_id"][i]])
            guess = predictions[u][test["question_id"][i]] >= 0.5
            if guess == test["is_correct"][i]:
                correct += 1
            total += 1

    return predictions_, correct / float(total)

if __name__ == "__main__":
    valid_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    print(k_means_category(8, valid_data)[1])