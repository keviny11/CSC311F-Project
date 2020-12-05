import numpy as np
import pandas as pd
from utils import *
from sklearn.cluster import KMeans

def prepare_data(matrix=True):

    """
    We are constructing a 3-d matrix, try feeding this into nn
    """
    train_data = load_train_csv("../data")
    train_m = load_train_sparse("../data")
    print(train_data.keys())

    data = np.empty((train_m.shape[0], train_m.shape[1], 388))
    data[:] = np.nan

    clster = 20 # change this
    q_meta_cat = q_meta_k_means(clster)

    for i, _ in enumerate(train_data["is_correct"]):
        uid, qid, correct = train_data["user_id"][i], \
                            train_data["question_id"][i], \
                            train_data["is_correct"][i]
        category = q_meta_cat[qid]
        data[uid][qid][category] = correct

    return data

def q_meta_k_means(clstr):
    q_meta_data = pd.read_csv("../data/question_meta.csv")
    q_meta_data = q_meta_data.sort_values("question_id")

    q_meta = np.zeros((q_meta_data.shape[0], 388))
    for index, row in q_meta_data.iterrows():
        for sid in eval(row["subject_id"]):
            q_meta[int(index), int(sid)] = 1

    kmeans = KMeans(n_clusters=clstr, random_state=0).fit(q_meta)
    return kmeans.labels_


if __name__ == "__main__":
    prepare_data()