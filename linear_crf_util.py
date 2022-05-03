from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
import numpy as np

################Evaluate crf models#########
class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))
def evaluate_batch_insts(batch_insts,
                         batch_pred_ids,
                         batch_gold_ids,
                         word_seq_lens,
                         idx2label):
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    batch_p_dict = defaultdict(int)
    batch_total_entity_dict = defaultdict(int)
    batch_total_predict_dict = defaultdict(int)

    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        
        #print(prediction)
        #convert to span
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("b-"):
                start = i
            if output[i].startswith("e-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
                batch_total_entity_dict[output[i][2:]] += 1
            if output[i].startswith("s-"):
                output_spans.add(Span(i, i, output[i][2:]))
                batch_total_entity_dict[output[i][2:]] += 1
        predict_spans = set()
        start = -1
        for i in range(len(prediction)):
            if prediction[i].startswith("b-"):
                start = i
            if prediction[i].startswith("e-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
                batch_total_predict_dict[prediction[i][2:]] += 1
            if prediction[i].startswith("s-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))
                batch_total_predict_dict[prediction[i][2:]] += 1

        correct_spans = predict_spans.intersection(output_spans)
        for span in correct_spans:
            batch_p_dict[span.type] += 1

    return Counter(batch_p_dict), Counter(batch_total_predict_dict), Counter(batch_total_entity_dict)

def get_metric(p_num: int, total_num: int, total_predicted_num: int) -> Tuple[float, float, float]:
    """
    Return the metrics of precision, recall and f-score, based on the number
    (We make this small piece of function in order to reduce the code effort and less possible to have typo error)
    :param p_num:
    :param total_num:
    :param total_predicted_num:
    :return:
    """
    precision = p_num * 1.0 / total_predicted_num * 100 if total_predicted_num != 0 else 0
    recall = p_num * 1.0 / total_num * 100 if total_num != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    return precision, recall, fscore

