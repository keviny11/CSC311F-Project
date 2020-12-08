import numpy as np
import pandas as pd
from utils import *
from sklearn.cluster import KMeans
import sys

def k_means_category(clster):

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

        # std = 0.5/3, 99.7% in [0, 1]
        guess = np.random.normal(loc=0.5, scale=0.5/3) # if we have no data, guess this
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

    kmeans = KMeans(n_clusters=clstr, random_state=0).fit(q_meta)
    return kmeans.labels_

def evaluate(predictions):
    valid_data = load_train_csv("../data")

    total, correct = 0, 0
    for i, u in enumerate(valid_data["user_id"]):

        guess = predictions[u][valid_data["question_id"][i]] >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)

if __name__ == "__main__":
    pred = k_means_category(342)
    acc = evaluate(pred)
    print(acc)