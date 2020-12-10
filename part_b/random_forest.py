from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils import *

def random_forest():
    train_data = load_train_csv("../data")
    train_matrix = load_train_sparse("../data").toarray()
    q_meta_data = pd.read_csv("../data/question_meta.csv")
    q_meta_data = q_meta_data.sort_values("question_id")

    q_meta = np.zeros((q_meta_data.shape[0], 388))
    for index, row in q_meta_data.iterrows():
        for sid in eval(row["subject_id"]):
            q_meta[int(index), int(sid)] = 1

    RF_train_dat = [[] for _ in range(train_matrix.shape[0])]
    RF_train_label = [[] for _ in range(train_matrix.shape[0])]

    for i, correct in enumerate(train_data["is_correct"]):
        uid, qid = train_data["user_id"][i], train_data["question_id"][i]
        RF_train_dat[uid].append(q_meta[qid])
        RF_train_label[uid].append(correct)

    RF_trees = []
    for i, X in enumerate(RF_train_dat):
        y = RF_train_label[i]
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)
        RF_trees.append(clf)

    evaluate_RF(RF_trees, q_meta)

def evaluate_RF(RF_trees, q_meta):
    valid_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    c = 0 # correct
    for i, correct in enumerate(valid_data["is_correct"]):
        uid, qid = valid_data["user_id"][i], valid_data["question_id"][i]

        if RF_trees[uid].predict([q_meta[qid]]) == correct:
            c += 1
    print(c/len(valid_data["is_correct"]))

if __name__ == "__main__":
    random_forest()
