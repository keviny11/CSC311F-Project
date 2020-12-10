import pandas as pd
from utils import *

def cosine_similarity(top=3):
    train_data = load_train_csv("../data")
    train_matrix = load_train_sparse("../data").toarray()
    q_meta_data = pd.read_csv("../data/question_meta.csv")
    q_meta_data = q_meta_data.sort_values("question_id")

    num_question = q_meta_data.shape[0]
    q_meta = np.zeros((num_question, 388))
    for index, row in q_meta_data.iterrows():
        for sid in eval(row["subject_id"]):
            q_meta[int(index), int(sid)] = 1

    rankings = [[] for _ in range(num_question)]
    for i in range(num_question):
        for j in range(num_question):
            angle = compute_angle(q_meta[i], q_meta[j])
            rankings[i].append(angle)
        rankings[i] = np.argsort(rankings[i])

    evaluate_CS(rankings, top)

def compute_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def evaluate_CS(rankings, top):
    valid_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    train_matrix = load_train_sparse("../data").toarray()

    c = 0
    for i, correct in enumerate(valid_data["is_correct"]):
        sum, count = 0, 0
        uid, qid = valid_data["user_id"][i], valid_data["question_id"][i]
        for r in rankings[uid]:
            if count == top:
                break
            if np.isnan(train_matrix[uid][r]):
                continue
            sum += train_matrix[uid][r]
            count += 1
        prediction = sum/top >= 0.5
        if prediction == correct:
            c += 1
    print(c/len(valid_data["is_correct"]))

if __name__ == "__main__":
    cosine_similarity(2)