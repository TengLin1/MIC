from typing import Iterable, Union, List

import numpy as np
from prenlp.data import IMDB

import torch
from torch.utils.data import TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from collections import OrderedDict

DATASETS_CLASSES = {'imdb': IMDB}


class InputExample:
    """A single training/test example for text classification.
    """

    def __init__(self, text: str, label: str):
        self.text = text
        self.label = label


class InputFeatures:
    """A single set of features of data.
    """

    def __init__(self, input_ids: List[int], label_id: int):
        self.input_ids = input_ids
        self.label_id = label_id


def convert_examples_to_features(examples: List[InputExample],
                                 n_l,
                                 label_dict: dict,
                                 tokenizer,
                                 max_seq_len: int) -> List[InputFeatures]:
    pad_token_id = tokenizer.pad_token_id
    features = []
    for i, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = tokens[:max_seq_len]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        label_id = label_dict.get(example.label[n_l])

        feature = InputFeatures(input_ids, label_id)  # float(example.label)
        features.append(feature)

    return features


def create_examples(args,
                    tokenizer, start, end, ant) -> Iterable[Union[List[InputExample], dict]]:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print("Reading input file")
    feature_inp = pd.read_csv(args.dataset, index_col=0)
    # genes after snp
    X_file = pd.read_csv("../data/snps.subtitude.data1.csv", index_col=0, sep='\t')
    print("input file is imported")
    X = X_file.iloc[:, list(range(0, X_file.shape[1]))]  # X_file.shape[1]
    X = X.values
    XGB_file = pd.read_csv("../data/XGB_importance_200.csv", index_col=0)
    # XGB_file = pd.read_csv("../data/att_weight_2000.csv", index_col=0)
    print("XGB file is imported")
    XGB = XGB_file.iloc[:, 1]  # X_file.shape[1]
    XGB = XGB.values.reshape(XGB.shape[0], 1)
    # print(XGB)
    Xxgb = np.hstack((X, XGB))
    Xxgb = pd.DataFrame(Xxgb).dropna().values
    Xxgb = Xxgb[:, :Xxgb.shape[1] - 1].T
    # columns = X.columns
    print(X.shape)
    # if ant == 0:
    #     y_list = list([8])#, 1, 2, 4, 5, 6, 7, 8, 9, 11, 13, 14
    # else:
    #     y_list = list([3, 12])#
    y_list = list([ant])
    y = feature_inp.iloc[:, y_list]  # list([0,1,2,3,5,6,7,8,9,11,13,14])y_list
    y = y.values.reshape(y.shape[0], len(y_list))
    Xy = np.hstack((y, Xxgb))
    print('Xy.shape =', Xy.shape)
    Xy = pd.DataFrame(Xy).dropna().values
    print('Xy.shape1 =', Xy.shape)
    y = Xy[:, 0:len(y_list)]  # Xy.shape[0] - 1
    X = Xy[:, len(y_list):]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    f_labels, weights = [], []
    train_examples = []
    train_all_label_ids = []
    # print(X_train[0].shape)

    ''' train data set'''
    for i in range(len(y_train)):
        text = ''.join(alle + ' ' for alle in X_train[i])
        text = '[CLS] ' + text
        print('X =', text)
        example = InputExample(text, y_train[i])
        train_examples.append(example)

    ''' test data set'''
    test_examples = []
    test_all_label_ids = []

    for i in range(len(y_test)):
        text = ''.join(alle + ' ' for alle in X_test[i])
        text = '[CLS] ' + text
        example = InputExample(text, y_test[i])
        test_examples.append(example)
    all_labelss_15 = [[32.0, 1.0, 2.0, 4.0, 16.0, 8.0],
                      [32.0, 16.0, 1.0, 2.0, 8.0, 4.0],
                      [0.25, 16.0, 32.0, 0.5, 1.0, 2.0, 4.0, 8.0, 64.0],
                      [1.0, 2.0, 4.0, 8.0, 16.0],
                      [2.0, 4.0, 8.0, 16.0, 32.0],
                      [0.015, 0.01, 0.03, 0.06, 0.12, 0.125, 0.25, 0.5, 1.0, 2.0],
                      [0.125, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0],
                      [16.0, 32.0, 256.0, 64.0, 128.0],
                      [32.0, 1.0, 2.0, 4.0, 8.0, 16.0],
                      [0.5, 0.25, 16.0, 1.0, 2.0, 4.0, 8.0],
                      [16.0, 32.0, 64.0, 8.0],
                      [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
                      [2.0, 4.0, 8.0, 64.0, 16.0, 32.0],
                      [4.0, 32.0, 8.0, 16.0],
                      [0.25, 0.5, 1.0, 8.0, 2.0, 4.0]]
    all_labelss = [all_labelss_15[ant]]
    # if ant == 0:
    #     all_labelss = [[32.0, 1.0, 2.0, 4.0, 16.0, 8.0],
    #                   [32.0, 16.0, 1.0, 2.0, 8.0, 4.0],
    #                   [0.25, 16.0, 32.0, 0.5, 1.0, 2.0, 4.0, 8.0, 64.0],
    #                   [2.0, 4.0, 8.0, 16.0, 32.0],
    #                   [0.015, 0.01, 0.03, 0.06, 0.12, 0.125, 0.25, 0.5, 1.0, 2.0],
    #                   [0.125, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0],
    #                   [16.0, 32.0, 256.0, 64.0, 128.0],
    #                   [32.0, 1.0, 2.0, 4.0, 8.0, 16.0],
    #                   [0.5, 0.25, 16.0, 1.0, 2.0, 4.0, 8.0],
    #                   [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    #                   [4.0, 32.0, 8.0, 16.0],
    #                   [0.25, 0.5, 1.0, 8.0, 2.0, 4.0]]
    # else:
    #     all_labelss = [[1.0, 2.0, 4.0, 8.0, 16.0], [2.0, 4.0, 8.0, 64.0, 16.0, 32.0]]

    for i in range(len(y_list)):
        train_label_dict = {label: k for k, label in enumerate(all_labelss[i])}
        test_label_dict = {label: k for k, label in enumerate(all_labelss[i])}
        train_features = convert_examples_to_features(train_examples, i, train_label_dict, tokenizer, args.max_seq_len)
        train_input_ids = torch.tensor([feature.input_ids for feature in train_features], dtype=torch.long)
        train_label_ids = np.array([feature.label_id for feature in train_features])
        train_all_label_ids.append(train_label_ids)

        test_features = convert_examples_to_features(test_examples, i, test_label_dict, tokenizer, args.max_seq_len)
        test_input_ids = torch.tensor([feature.input_ids for feature in test_features], dtype=torch.long)
        test_label_ids = np.array([feature.label_id for feature in test_features])
        # print(test_label_ids)
        test_all_label_ids.append(test_label_ids)

        all_label_ids = np.hstack((test_label_ids, train_label_ids))
        # labels = train_labels + test_labels
        f_label = sorted(list(set([n for n in all_labelss[i]])))
        # f_label = list(set([n for n in all_labels]))
        # print(np.unique(all_label_ids))
        weight = compute_class_weight('balanced', np.unique(all_label_ids),
                                      all_label_ids).tolist()
        # weight = compute_class_weight('balanced', np.array(range(0, len(f_label))),
        #                               all_label_ids).tolist()
        f_labels.append(f_label)
        weights.append(weight)
        # weights = weight

    train_all_label_ids = torch.tensor(train_all_label_ids, dtype=torch.long)
    test_all_label_ids = torch.tensor(test_all_label_ids, dtype=torch.long)
    train_dataset = TensorDataset(train_input_ids, train_all_label_ids.t())
    test_dataset = TensorDataset(test_input_ids, test_all_label_ids.t())
    print('f_labels =', f_labels)
    # print('weights =', weights)
    return train_dataset, test_dataset, f_labels, weights
