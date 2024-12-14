import json
import pandas as pd
import re
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
import copy
import nltk
from rouge_score import rouge_scorer


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

    df["acc_" + pred_col] = acc_list
    df["hit" + pred_col] = hit_list
    df["f1" + pred_col] = f1_list
    df["precision" + pred_col] = precission_list
    df["recall" + pred_col] = recall_list

    df.to_csv(path, index=False)

    print(f"Accuracy: {acc:.4f}")
    print(f"Hit: {hit:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return hit


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
        # return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction_str, ground_truth_str):
    bool1 = normalize_answer(prediction_str) == normalize_answer(ground_truth_str)
    bool2 = normalize(prediction_str) == normalize(ground_truth_str)
    return 1 if bool1 or bool2 else 0


def update_answer(prediction_str, answer_str):
    em = exact_match_score(prediction_str, answer_str)
    f1, precision, recall = f1_score(prediction_str, answer_str)
    return em, f1, precision, recall


def get_accuracy_doc_qa(path, pred_col="pred", label_col="label"):
    df = pd.read_csv(path, na_filter=False)

    # Load results
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    em_list = []

    label_list, pred_list = get_label_pred_list(df, pred_col, label_col)
    for prediction, answer in zip(pred_list, label_list):

        prediction = prediction.replace("|", "\n")
        prediction = prediction.split("\n")
        prediction_str = " ".join(prediction)

        answer = answer.split("|")
        if isinstance(answer, list):
            answer_str = " ".join(answer)
        else:
            answer_str = answer

        hit = eval_hit(prediction_str, answer)
        hit_list.append(hit)

        em, f1, prec, recall = update_answer(prediction_str, answer_str)
        em_list.append(em)
        f1_list.append(f1)
        precission_list.append(prec)
        recall_list.append(recall)

    hit = sum(hit_list) * 100 / len(hit_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    recall = sum(recall_list) * 100 / len(recall_list)
    em = sum(em_list) * 100 / len(em_list)

    df["hit" + pred_col] = hit_list
    df["f1" + pred_col] = f1_list
    df["precision" + pred_col] = precission_list
    df["recall" + pred_col] = recall_list
    df["em" + pred_col] = em_list

    df.to_csv(path, index=False)

    print(f"Hit: {hit:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"EM: {em:.4f}")

    return hit


""" Evaluation script for NarrativeQA dataset. (Extracted from the official evaluation script) """

nltk_path = "/mnt/data/wangshu/hcarag/nltk"
# 添加 NLTK 数据路径
nltk.data.path.append(nltk_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Downloading punkt")
    nltk.download("punkt", download_dir=nltk_path)
    nltk.download("wordnet", download_dir=nltk_path)


def bleu_1(p, g):
    return sentence_bleu(g, p, weights=(1, 0, 0, 0))


def bleu_4(p, g):
    return sentence_bleu(g, p, weights=(0, 0, 0, 1))


def bleu_4_modify(p, g):
    return sentence_bleu(g, p, weights=(0.25, 0.25, 0.25, 0.25))


def bleu_1_smooth(p, g):
    return sentence_bleu(
        g, p, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1
    )


def bleu_4_smooth(p, g):
    return sentence_bleu(
        g, p, weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method1
    )


def bleu_4_modify_smooth(p, g):
    return sentence_bleu(
        g,
        p,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )


def meteor(p, g):
    return meteor_score([x.split() for x in g], p.split())


# 创建 RougeScorer 实例，设置 ROUGE-L 指标
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# 计算 ROUGE-L 分数
def rouge_l(p, g):
    if isinstance(g, list):
        g = g[0]

    return scorer.score(g, p)  # g: ground truth, p: prediction


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, tokenize=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if tokenize:
            score = metric_fn(word_tokenize(prediction), [word_tokenize(ground_truth)])
        else:
            score = metric_fn(prediction, [ground_truth])
        scores_for_ground_truths.append(score)
    if isinstance(score, dict) and "rougeL" in score:
        rouge_l_score = {"rouge_l f1": 0, "rouge_l precision": 0, "rouge_l recall": 0}
        rouge_l_score["rouge_l f1"] = max(
            [score["rougeL"].fmeasure for score in scores_for_ground_truths]
        )
        rouge_l_score["rouge_l precision"] = max(
            [score["rougeL"].precision for score in scores_for_ground_truths]
        )
        rouge_l_score["rouge_l recall"] = max(
            [score["rougeL"].recall for score in scores_for_ground_truths]
        )

        return rouge_l_score
    else:
        return round(max(scores_for_ground_truths), 2)


def get_metric_score(prediction, ground_truths):
    bleu_1_score = metric_max_over_ground_truths(
        bleu_1, prediction, ground_truths, tokenize=True
    )
    bleu_4_score = metric_max_over_ground_truths(
        bleu_4, prediction, ground_truths, tokenize=True
    )
    modify_bleu_4_score = metric_max_over_ground_truths(
        bleu_4_modify, prediction, ground_truths, tokenize=True
    )
    bleu_1_smooth_score = metric_max_over_ground_truths(
        bleu_1_smooth, prediction, ground_truths, tokenize=True
    )
    bleu_4_smooth_score = metric_max_over_ground_truths(
        bleu_4_smooth, prediction, ground_truths, tokenize=True
    )
    modify_bleu_4_smooth_score = metric_max_over_ground_truths(
        bleu_4_modify_smooth, prediction, ground_truths, tokenize=True
    )
    meteor_score = metric_max_over_ground_truths(
        meteor, prediction, ground_truths, tokenize=False
    )
    rouge_l_score = metric_max_over_ground_truths(
        rouge_l, prediction, ground_truths, tokenize=False
    )

    return {
        "bleu_1": bleu_1_score,
        "bleu_4": bleu_4_score,
        "modify_bleu_4": modify_bleu_4_score,
        "bleu_1_smooth": bleu_1_smooth_score,
        "bleu_4_smooth": bleu_4_smooth_score,
        "modify_bleu_4_smooth": modify_bleu_4_smooth_score,
        "meteor": meteor_score,
        "rouge_l f1": rouge_l_score["rouge_l f1"],
        "rouge_l precision": rouge_l_score["rouge_l precision"],
        "rouge_l recall": rouge_l_score["rouge_l recall"],
    }


def get_bleu_doc_qa(path, pred_col="pred", label_col="label"):
    df = pd.read_csv(path, na_filter=False)

    label_list, pred_list = get_label_pred_list(df, pred_col, label_col)

    # Load results
    bleu_1_list = []
    bleu_4_list = []
    modify_bleu_4_list = []
    bleu_1_smooth_list = []
    bleu_4_smooth_list = []
    modify_bleu_4_smooth_list = []
    meteor_list = []
    rouge_l_f1_list = []
    rouge_l_precision_list = []
    rouge_l_recall_list = []

    for prediction, answer in zip(pred_list, label_list):

        prediction = prediction.replace("|", "\n")
        prediction = prediction.split("\n")
        prediction_str = " ".join(prediction)

        answer = answer.split("|")

        metrics_res = get_metric_score(prediction_str, answer)
        bleu_1_list.append(metrics_res["bleu_1"])
        bleu_4_list.append(metrics_res["bleu_4"])
        modify_bleu_4_list.append(metrics_res["modify_bleu_4"])
        bleu_1_smooth_list.append(metrics_res["bleu_1_smooth"])
        bleu_4_smooth_list.append(metrics_res["bleu_4_smooth"])
        modify_bleu_4_smooth_list.append(metrics_res["modify_bleu_4_smooth"])
        meteor_list.append(metrics_res["meteor"])
        rouge_l_f1_list.append(metrics_res["rouge_l f1"])
        rouge_l_precision_list.append(metrics_res["rouge_l precision"])
        rouge_l_recall_list.append(metrics_res["rouge_l recall"])

    bleu_1 = sum(bleu_1_list) * 100 / len(bleu_1_list)
    bleu_4 = sum(bleu_4_list) * 100 / len(bleu_4_list)
    modify_bleu_4 = sum(modify_bleu_4_list) * 100 / len(modify_bleu_4_list)
    bleu_1_smooth = sum(bleu_1_smooth_list) * 100 / len(bleu_1_smooth_list)
    bleu_4_smooth = sum(bleu_4_smooth_list) * 100 / len(bleu_4_smooth_list)
    modify_bleu_4_smooth = (
        sum(modify_bleu_4_smooth_list) * 100 / len(modify_bleu_4_smooth_list)
    )
    meteor = sum(meteor_list) * 100 / len(meteor_list)
    rouge_l_f1 = sum(rouge_l_f1_list) * 100 / len(rouge_l_f1_list)
    rouge_l_precision = sum(rouge_l_precision_list) * 100 / len(rouge_l_precision_list)
    rouge_l_recall = sum(rouge_l_recall_list) * 100 / len(rouge_l_recall_list)

    df["bleu_1" + pred_col] = bleu_1_list
    df["bleu_4" + pred_col] = bleu_4_list
    df["modify_bleu_4" + pred_col] = modify_bleu_4_list
    df["bleu_1_smooth" + pred_col] = bleu_1_smooth_list
    df["bleu_4_smooth" + pred_col] = bleu_4_smooth_list
    df["modify_bleu_4_smooth" + pred_col] = modify_bleu_4_smooth_list
    df["meteor" + pred_col] = meteor_list
    df["rouge_l_f1" + pred_col] = rouge_l_f1_list
    df["rouge_l_precision" + pred_col] = rouge_l_precision_list
    df["rouge_l_recall" + pred_col] = rouge_l_recall_list

    df.to_csv(path, index=False)

    print(f"Bleu-1: {bleu_1:.4f}")
    print(f"Bleu-4: {bleu_4:.4f}")
    print(f"Modify Bleu-4: {modify_bleu_4:.4f}")
    print(f"Bleu-1 Smooth: {bleu_1_smooth:.4f}")
    print(f"Bleu-4 Smooth: {bleu_4_smooth:.4f}")
    print(f"Modify Bleu-4 Smooth: {modify_bleu_4_smooth:.4f}")
    print(f"Meteor: {meteor:.4f}")
    print(f"Rouge-l F1: {rouge_l_f1:.4f}")
    print(f"Rouge-l Precision: {rouge_l_precision:.4f}")
    print(f"Rouge-l Recall: {rouge_l_recall:.4f}")

    return bleu_1


dataset_name_path = {
    "webq": "/mnt/data/wangshu/hcarag/FB15k/webqa/webqa.json",
    "mintaka": "/mnt/data/wangshu/hcarag/mintaka/QA/mintaka_test_qa.json",
    "multihop": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/MultiHopRAG_qa.json",
    "hotpot": "/mnt/data/wangshu/hcarag/HotpotQA/dataset/eval_hotpot_qa.json",
    "narrativeqa": "/mnt/data/wangshu/hcarag/narrativeqa/dataset/narrativeqa_all.json",
    "narrativeqa_train":"/mnt/data/wangshu/hcarag/narrativeqa/data/train/{doc_idx}/qa_dataset/narrativeqa.json",
    "narrativeqa_test":"/mnt/data/wangshu/hcarag/narrativeqa/data/test/{doc_idx}/qa_dataset/narrativeqa.json",
    "webqsp": "/mnt/data/wangshu/hcarag/WebQSP/dataset/webqsp_qa.json",
}

baseline_save_path_dict = {
    "mintaka": "/mnt/data/wangshu/hcarag/mintaka/QA/baseline",
    "webq": "/mnt/data/wangshu/hcarag/FB15k/webqa/baseline",
    "multihop": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/baseline",
    "hotpot": "/mnt/data/wangshu/hcarag/HotpotQA/dataset/baseline",
    "narrativeqa": "/mnt/data/wangshu/hcarag/narrativeqa/dataset/baseline",
    "narrativeqa_train":"/mnt/data/wangshu/hcarag/narrativeqa/data/train/{doc_idx}/qa_dataset",
    "narrativeqa_test":"/mnt/data/wangshu/hcarag/narrativeqa/data/test/{doc_idx}/qa_dataset",
    "webqsp": "/mnt/data/wangshu/hcarag/WebQSP/dataset/baseline",
}


if __name__ == "__main__":
    save_file_qa = (
        "/mnt/data/wangshu/hcarag/FB15k/hc_index_8b/qa/global_5_10_15_direct_QA_.json"
    )
    acc_raw = get_accuracy_webqsp_qa(save_file_qa, pred_col="raw_result")
