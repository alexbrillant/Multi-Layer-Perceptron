from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from imblearn.metrics import classification_report_imbalanced


def precision_score_weighted(data_inputs, expected_outputs):
    return precision_score(np.argmax(data_inputs, axis=1), np.argmax(expected_outputs, axis=1), average='weighted')


def recall_score_weighted(data_inputs, expected_outputs):
    return recall_score(np.argmax(data_inputs, axis=1), np.argmax(expected_outputs, axis=1), average='weighted')


def f1_score_weighted(data_inputs, expected_outputs):
    return f1_score(np.argmax(data_inputs, axis=1), np.argmax(expected_outputs, axis=1), average='weighted')

def classificaiton_report_imbalanced_metric(data_inputs, expected_outputs):
    return '\n'+ classification_report_imbalanced(np.argmax(data_inputs, axis=1), np.argmax(expected_outputs, axis=1))
