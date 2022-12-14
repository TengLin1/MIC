
import numpy as np
from tqdm import tqdm
import infomap
import time
from multiprocessing.dummy import Pool as Threadpool
from multiprocessing import Pool
import multiprocessing as mp
import os
from utils import Timer
from evaluation import evaluate, accuracy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# %matplotlib inline

def l2norm(vec):
    """
    归一化
    :param vec: 
    :return: 
    """
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    print("print(d.items()) =", d.items())
    for k, v in d.items():

        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_faiss(knn):
    """
    内积暴力循环
    归一化特征的内积等价于余弦相似度
    """
    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 knn_method='faiss-cpu',
                 verbose=True):
        import faiss
        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                print('[{}] read knns from {}'.format(knn_method, knn_ofn))
                self.knns = np.load(knn_ofn)['data']
            else:
                feats = feats.astype('float32')
                size, dim = feats.shape
                if knn_method == 'faiss-gpu':
                    import math
                    i = math.ceil(size/1000000)
                    if i > 1:
                        i = (i-1)*4
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(i * 1024 * 1024 * 1024)
                    index = faiss.GpuIndexFlatIP(res, dim)
                else:
                    index = faiss.IndexFlatIP(dim)
                index.add(feats)
        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                pass
            else:
                sims, nbrs = index.search(feats, k=k)
                # torch.cuda.empty_cache()
                self.knns = [(np.array(nbr, dtype=np.int32),
                              1 - np.array(sim, dtype=np.float32))
                             for nbr, sim in zip(nbrs, sims)]


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# 构造边
def get_links(single, links, nbrs, dists):
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(0, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - min_sim:
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
            else:
                break
        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links


def cluster_by_infomap(G, nbrs, dists, pred_label_path, save_result=False):
    """
    基于infomap的聚类
    :param nbrs: 
    :param dists: 
    :param pred_label_path: 
    :return: 
    """
    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists)

    infomapWrapper = infomap.Infomap("--two-level --directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    # 聚类运算
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    # 聚类结果统计
    for node in infomapWrapper.iterTree():
        # node.physicalId 特征向量的编号
        # node.moduleIndex() 聚类的编号
        # print(node)
        idx2label[node.physicalId] = node.moduleIndex()
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)
    # print(idx2label)
    # pos = nx.spring_layout(G)
    # print(pos)
    # nx.set_node_attributes(G, name='community', values=idx2label)
    # communities = [v for k, v in idx2label.items()]
    # numCommunities = max(communities) + 1
    # # color map from http://colorbrewer2.org/
    # cmapLight = colors.ListedColormap(
    #     ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#1f78b4', '#00FFFF', '#E0FFFF', '#5F9EA0', '#FF3E96',
    #      '#473C8B'], 'indexed', numCommunities)
    # cmapDark = colors.ListedColormap(['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a'], 'indexed', numCommunities)
    #
    # # Draw edges
    # nx.draw_networkx_edges(G, pos)
    #
    # # Draw nodes
    # nodeCollection = nx.draw_networkx_nodes(G,
    #                                         pos=pos,
    #                                         node_color=communities,
    #                                         cmap=cmapLight
    #                                         )
    # # Set node border color to the darker shade
    # darkColors = [cmapDark(v) for v in communities]
    # nodeCollection.set_edgecolor(darkColors)
    #
    # # Draw node labels
    # for n in G.nodes():
    #     plt.annotate(n,
    #                  xy=pos[n],
    #                  textcoords='offset points',
    #                  horizontalalignment='center',
    #                  verticalalignment='center',
    #                  xytext=[0, 0],
    #                  color=cmapDark(communities[n])
    #                  )
    #
    # plt.axis('off')
    # # plt.savefig("karate.png")
    # plt.show()

    node_count = 0
    for k, v in label2idx.items():
        if k == 0:
            node_count += len(v[2:])
            label2idx[k] = v[2:]
            # print(k, v[2:])
        else:
            node_count += len(v[1:])
            label2idx[k] = v[1:]
            # print(k, v[1:])

    # print(node_count)
    # 孤立点个数
    print("孤立点数：{}".format(len(single)))

    keys_len = len(list(label2idx.keys()))
    # print(keys_len)

    # 孤立点放入到结果中
    for single_node in single:
        idx2label[single_node] = keys_len
        label2idx[keys_len] = [single_node]
        keys_len += 1

    print("总类别数：{}".format(keys_len))

    idx_len = len(list(idx2label.keys()))
    print("总节点数：{}".format(idx_len))

    # 保存结果
    if save_result:
        with open(pred_label_path, 'w') as of:
            for idx in range(idx_len):
                of.write(str(idx2label[idx]) + '\n')

    if label_path is not None:
        pred_labels = intdict2ndarray(idx2label)
        # print("pred_labels = ", pred_labels)
        # true_lb2idxs, true_idx2lb = read_meta(label_path)
        label_file = pd.read_csv(label_path, index_col=0)
        true_idx2lb = label_file.iloc[:, 0].values
        # print("true_idx2lb = ", true_idx2lb)
        gt_labels = intdict2ndarray(true_idx2lb)
        print("gt_labels = ", gt_labels.shape)
        for metric in metrics:
            evaluate(gt_labels, pred_labels, metric)


def get_dist_nbr(feature_path, k=80, knn_method='faiss-cpu'):
    # features = np.fromfile(feature_path, dtype=np.float32)
    features_file = pd.read_csv(feature_path, index_col=0)
    features = features_file.iloc[:, 0].values
    # print(features.shape)
    features = features.reshape(-1, 32)
    features = l2norm(features)

    index = knn_faiss(feats=features, k=k, knn_method=knn_method)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    return dists, nbrs

def drawNetwork(G):
    # position map
    pos = nx.spring_layout(G)
    print(pos)
    # community ids
    communities = [v for k,v in nx.get_node_attributes(G, 'community').items()]
    print(communities)
    numCommunities = max(communities) + 1
    # color map from http://colorbrewer2.org/
    cmapLight = colors.ListedColormap(['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#1f78b4', '#00FFFF', '#E0FFFF', '#5F9EA0', '#FF3E96', '#473C8B'], 'indexed', numCommunities)
    cmapDark = colors.ListedColormap(['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a'], 'indexed', numCommunities)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw nodes
    nodeCollection = nx.draw_networkx_nodes(G,
        pos = pos,
        node_color = communities,
        cmap = cmapLight
    )
    # Set node border color to the darker shade
    darkColors = [cmapDark(v) for v in communities]
    nodeCollection.set_edgecolor(darkColors)

    # Draw node labels
    for n in G.nodes():
        plt.annotate(n,
            xy = pos[n],
            textcoords = 'offset points',
            horizontalalignment = 'center',
            verticalalignment = 'center',
            xytext = [0, 0],
            color = cmapDark(communities[n])
        )

    plt.axis('off')
    # plt.savefig("karate.png")
    plt.show()

knn_method = 'faiss-cpu'
metrics = ['pairwise', 'bcubed', 'nmi']
min_sim = 0.58
k = 50
# true_label
label_path = 'data/cluster_data/100_test_ant0-summary_ground_test.csv'
feature_path = 'data/cluster_data/100_test_ant0-summary_feature_test.csv'
pred_label_path = 'data/cluster_data/test_predict.txt'
graph = nx.karate_club_graph()

with Timer('All face cluster step'):
    dists, nbrs = get_dist_nbr(feature_path=feature_path, k=k, knn_method=knn_method)
    # print(dists)
    print(nbrs)
    cluster_by_infomap(graph, nbrs, dists, pred_label_path, save_result=True)
    # drawNetwork(graph)


