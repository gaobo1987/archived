"""following the evaluation script of DrQA"""

from collections import Counter
import string
import re
from typing import List


def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> tuple:
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def reader_f1_max(pred_answer: str, gold_answers: List[str]) -> tuple:
    fs = []
    ps = []
    rs = []
    for gold_answer in gold_answers:
        f, p, r = f1_score(pred_answer, gold_answer)
        fs.append(f)
        ps.append(p)
        rs.append(r)
    return max(fs), max(ps), max(rs)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def regex_match_score(prediction: str, pattern: str) -> float:
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException as e:
        print('Regular expression failed to compile: %s' % pattern)
        print(e)
        return False
    return float(compiled.match(prediction) is not None)


def reader_match_max(metric_fn, pred_answer: str, gold_answers: List[str]) -> float:
    scores_for_ground_truths = []
    for gold_answer in gold_answers:
        score = metric_fn(pred_answer, gold_answer)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def retrieval_ranks(answer: str, retrieved_documents: List[str]) -> List[int]:
    """
    :param answer:
    :param retrieved_documents:
    :return: a list of ranks of which the retrieved documents contain the given answer (gold answer)
    """
    ranks = []
    answer_n = normalize_text(answer)
    for rank, d in enumerate(retrieved_documents):
        d = normalize_text(d)
        if answer_n in d:
            ranks.append(rank)
    return ranks


def retrieval_ranks_merge(answers: List[str], retrieved_documents: List[str]) -> List[int]:
    """
    :param answers: typically a set of gold answers to the same question
    :param retrieved_documents:
    :return: a merged list of ranks of which the retrieved documents contain the given answers
    """
    if not answers:
        raise Exception('Incorrect format of the gold_answers parameter')
    ranks = []
    for a in answers:
        ranks += retrieval_ranks(a, retrieved_documents)
    return list(set(ranks))


def retrieval_ranks_convert(target_ranks: List[int], reference_ranks: List[int]) -> List[int]:
    """
    This function convert the target_ranks to a new list of ranks according to the reference_ranks.

    More specifically, we have
    the documents with indices [0, 1, 2, 3, 4, 5], after reranking by e.g. Nearest Neighbor search of FAISS,
    we have the documents reranked according to their indices: [4, 3, 0, 1, 2, 5] (reference ranks),
    which means the document originally ranked at 4 is now ranks at 0,
    the document originally ranked at 3 now ranks at 1, and so on. We can list them as:
    4 -> 0
    3 -> 1
    0 -> 2
    1 -> 3
    2 -> 4
    5 -> 5

    Now suppose we have a list of target ranks that still follows the original ranks numbers, say
    [0, 3, 5], and we want to convert it according to the reference ranks, then we would have: [2, 1, 5]
    :param target_ranks:
    :param reference_ranks:
    :return:
    """
    rank_map = {}
    for new_rank, old_rank in enumerate(reference_ranks):
        rank_map[str(old_rank)] = new_rank
    result_ranks = []
    for old_rank in target_ranks:
        new_rank = rank_map[str(old_rank)]
        result_ranks.append(new_rank)
    return result_ranks


def recall_at_k(ranks_list: List[List[int]], k:int) -> float:
    """
    This function evaluates the recall of a retriever at rank K:
    the percentage of questions of which the answers appear in top K retrieved documents.
    :param ranks_list: each element corresponds to the ranks of the gold answer(s) of a question
    :param k: the rank threshold
    :return:
    """
    if ranks_list:
        total = len(ranks_list)
        hits = 0
        for ranks in ranks_list:
            if ranks and min(ranks) < k:
                hits += 1

        return float(hits) / total
    return 0


def precision_at_k(ranks_list: List[List[int]], k:int) -> float:
    """
    This function evaluates the precision of a retriever at rank k:
    the average percentage of documents in which the answer of a question appears, out of top K retrieved documents
    :param ranks_list:
    :param k:
    :return:
    """
    if ranks_list and k > 0:
        precisions = []
        for ranks in ranks_list:
            hits = 0
            for r in ranks:
                if r < k:
                    hits += 1
            p = hits / float(k)
            precisions.append(p)
        return sum(precisions) / len(precisions)
    return 0
