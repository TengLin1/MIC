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


def create_examples(args, ant) -> Iterable[Union[List[InputExample], dict]]:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from xgboost import plot_importance

    print("Reading input file")
    feature_inp = pd.read_csv(args.dataset, index_col=0)
    # feature_inp = feature_inp.sort_values(by='strains', ascending=False)

    # genes after snp
    X_file = pd.read_csv("../data/snps.subtitude.data1.csv", index_col=0, sep='\t')
    print("input file is imported")
    X = X_file.iloc[:, list(range(0, X_file.shape[1]))]  # X_file.shape[1]
    X = X.values
    # XGB_file = pd.read_csv("data/XGB_importance_class_" + str(ant) + ".csv", index_col=0)
    # XGB_file = pd.read_csv("../data/XGB_importance_" + str(args.max_seq_len) + ".csv", index_col=0)
    # XGB_file = pd.read_csv("../data/arg_sites.csv", index_col=0)
    XGB_file = pd.read_csv("../data/XGB_importance_class_" + str(ant) + ".csv", index_col=0)
    print("XGB file is imported")
    XGB = XGB_file.iloc[:, 1]  # X_file.shape[1]
    columns = XGB_file.iloc[:, 0].values
    XGB = XGB.values.reshape(XGB.shape[0], 1)
    Xxgb = np.hstack((X, XGB))
    Xxgb = pd.DataFrame(Xxgb).dropna().values
    Xxgb = Xxgb[:, :Xxgb.shape[1] - 1].T

    y_all = list([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14])

    y_list_cluster = [y_all[ant]]# 15 is for cluster
    y_list = [y_all[ant]]
    print('y_list =', y_list)
    y = feature_inp.iloc[:, y_list_cluster]  # list([0,1,2,3,5,6,7,8,9,11,13,14])y_list
    y = y.values.reshape(y.shape[0], len(y_list_cluster))
    Xy = np.hstack((y, Xxgb))
    print('Xy.shape =', Xy.shape)
    Xy = pd.DataFrame(Xy).dropna().values
    print('Xy.shape1 =', Xy.shape)
    y = Xy[:2000, 0:len(y_list)]  # Xy.shape[0] - 1
    X = Xy[:2000, len(y_list_cluster):]
    print('X.shape =', X.shape)

    encoding_seq = OrderedDict([
        ('UNK', [0, 0, 0, 0]),
        ('A', [1, 0, 0, 0]),
        ('C', [0, 1, 0, 0]),
        ('G', [0, 0, 1, 0]),
        ('T', [0, 0, 0, 1]),
        ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
    ])
    #
    # all_labelss = [[1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    #                [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    #                [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    #                [1.0, 2.0, 4.0, 8.0, 16.0],
    #                [2.0, 4.0, 8.0, 16.0, 32.0],
    #                [0.01, 0.015, 0.03, 0.06, 0.12, 0.125, 0.25, 0.5, 1.0, 2.0],
    #                [0.12, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
    #                [16.0, 32.0, 64.0, 128.0, 256.0],
    #                [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    #                [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    #                [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    #                [2.0, 4.0, 8.0, 64.0, 16.0, 32.0],
    #                [4.0, 8.0, 16.0, 32.0],
    #

    all_labelss = [[32.0, 1.0, 2.0, 4.0, 16.0, 8.0],
                   [32.0, 16.0, 1.0, 2.0, 8.0, 4.0],
                   [0.25, 16.0, 32.0, 0.5, 1.0, 2.0, 4.0, 8.0, 64.0],
                   [1.0, 2.0, 4.0, 8.0, 16.0],
                   [2.0, 4.0, 8.0, 16.0, 32.0],
                   [0.015, 0.01, 0.03, 0.06, 0.12, 0.125, 0.25, 0.5, 1.0, 2.0],
                   [0.125, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0],
                   [16.0, 32.0, 256.0, 64.0, 128.0],
                   [32.0, 1.0, 2.0, 4.0, 8.0, 16.0],
                   [0.5, 0.25, 16.0, 1.0, 2.0, 4.0, 8.0],
                   [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
                   [2.0, 4.0, 8.0, 64.0, 16.0, 32.0],
                   [4.0, 32.0, 8.0, 16.0],
                   [0.25, 0.5, 1.0, 8.0, 2.0, 4.0]]

    seq_encoding_keys = list(encoding_seq.keys())
    print('seq_encoding_keys = ', seq_encoding_keys)

    X_float = np.array([[seq_encoding_keys.index(c.upper()) for c in gene] for gene in X])
    y_float = np.array([all_labelss[ant].index(label) for label in y])
    print('y_float = ', y_float)

    X_train, X_test, y_train, y_test = train_test_split(X_float, y_float, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True,
                              objective='multi:softmax', eval_metric='mlogloss', scale_pos_weight=1)
    # model = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=500, silent=True,
    #                          objective='reg:linear', eval_metric='mae', scale_pos_weight=1)
    model.fit(X_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(X_test)
    print(y_pred)

    imlist = list(zip(list(columns), model.feature_importances_))
    imreport = pd.DataFrame(imlist, columns=['features', 'importance'])
    imreport.to_csv('results/single_XGB_importance_class_' + str(ant) + '_label—2000.csv', sep='\t')
    y_c = y_pred.astype('int')
    sum = 0
    sum_raw = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            sum_raw += 1

        if y_pred[i] >= y_test[i] -2 and y_pred[i] <= y_test[i] + 2:
            # print('pred_r = ', y_pred[i])
            # print('label = ', y_test[i])
            y_c[i] = 1
            sum += 1
        else:
            # print('pred_w = ', y_pred[i])
            # print('label = ', y_test[i])
            y_c[i] = 0
    summary = []
    summary.append([sum_raw / len(y_pred), sum / len(y_pred)])
    print(columns, sum / len(y_pred))
    report = list(zip(y_test, y_pred, y_c))
    report = pd.DataFrame(report, columns=['test', 'pred', 'correct or not'])
    report.to_csv('results/single_XGB_class_' + str(ant) + '_label—2000.csv', sep='\t')
    summary = pd.DataFrame(summary, columns=['acc-raw', 'acc'])
    summary.to_csv('results/single_XGB_class_summary_' + str(ant) + '_label—2000.csv', sep='\t')

