import pandas as pd
from utils import *
from scipy import stats

def cosine_similarity_subjects(top=3):
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

    evaluate_CS_subject(rankings, top)

def cosine_similarity_students(threshold=1.0, top=None):
    train_matrix = load_train_sparse("../data").toarray()
    train_matrix[train_matrix == 0] = -1
    np.nan_to_num(train_matrix, copy=False, nan=0)
    num_students = train_matrix.shape[0]

    rankings = [[] for _ in range(num_students)]
    valid = 0
    for i in range(num_students):
        for j in range(num_students):
            angle = compute_angle(train_matrix[i], train_matrix[j])
            rankings[i].append(angle)
            if i != j and angle < threshold:
                valid += 1
        rankings[i] = np.argsort(rankings[i])[:valid]

    test_data = load_public_test_csv("../data")
    train_matrix = load_train_sparse("../data").toarray()
    predictions = []

    for i, t in enumerate(test_data["is_correct"]):
        uid, qid = test_data["user_id"][i], test_data["question_id"][i]
        top = len(rankings[uid]) if top is None else min(len(rankings[uid]), top)

        if top > 0:
            total = 0
            for j in range(top):
                uid_ = rankings[uid][j]
                total += train_matrix[uid_][qid]
            predictions.append(total/top)
        else:
            predictions.append(None)

    return predictions

def compute_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def evaluate_CS_students(predictions):
    test_data = load_public_test_csv("../data")
    correct, total = 0, 0
    for i, t in enumerate(test_data["is_correct"]):
        if predictions[i] is not None:
            guess = predictions[i] >= 0.5
            if guess == t:
                correct += 1
            total += 1
    print(correct / total)

def evaluate_CS_subject(rankings, top):
    valid_data = load_valid_csv("../data")
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
    evaluate_CS_students(cosine_similarity_students(top=2))
    # cosine_similarity_subjects()