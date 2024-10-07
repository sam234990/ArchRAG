import json
import pandas as pd
import re
import string


def get_accuracy_gqa(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        if label in pred:
            correct += 1
    return correct / len(df)


def get_accuracy_expla_graphs(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        if len(matches) > 0 and matches[0].lower() == label:
            correct += 1

    return correct / len(df)


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    if precision > 1:
        precision = 1
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def get_label_pred_list(df, pred_col, label_col):
    label_list = df[label_col].tolist()
    pred_list = df[pred_col].tolist()
    return label_list, pred_list


def get_accuracy_webqsp_qa(path, pred_col="pred", label_col="label"):
    df = pd.read_csv(path, na_filter=False)

    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []

    label_list, pred_list = get_label_pred_list(df, pred_col, label_col)

    for prediction, answer in zip(pred_list, label_list):

        prediction = prediction.replace("|", "\n")
        answer = answer.split("|")

        prediction = prediction.split("\n")
        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        prediction_str = " ".join(prediction)
        acc = eval_acc(prediction_str, answer)
        hit = eval_hit(prediction_str, answer)
        acc_list.append(acc)
        hit_list.append(hit)

    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    recall = sum(recall_list) * 100 / len(recall_list)

    print(f"Accuracy: {acc:.4f}")
    print(f"Hit: {hit:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return hit


eval_funcs = {
    "expla_graphs": get_accuracy_expla_graphs,
    "scene_graphs": get_accuracy_gqa,
    "scene_graphs_baseline": get_accuracy_gqa,
    "webqsp": get_accuracy_webqsp_qa,
    "webqsp_baseline": get_accuracy_webqsp_qa,
}

if __name__ == "__main__":
    save_file_qa = (
        "/mnt/data/wangshu/hcarag/FB15k/hc_index_8b/qa/global_5_10_15_direct_QA_.json"
    )
    acc_raw = get_accuracy_webqsp_qa(save_file_qa, pred_col="raw_result")
