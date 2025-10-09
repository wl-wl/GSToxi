# import numpy as np
# x_pretrain = np.load("/tmp/pycharm_project_AMPMIC/data/EC_data_pretrain/EC_X_test10_prott5_pooled.npy")  # shape: [N, 1024]
# print(x_pretrain.shape)
# print(x_pretrain)
#这里代码有问题
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef, confusion_matrix
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import torch.nn as nn
from model import SimpleSelfAttention, Model_TGCN, batch_size,Transformer_test
from data_process import func
import numpy as np
import pandas as pd
import torch.nn.functional as F
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import dgl
from dgllife.utils import *
from dgllife.utils import smiles_to_bigraph
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
import umap
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, confusion_matrix
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def info_nce_loss(features):

    labels = torch.cat([torch.arange(int(features.shape[0]/2)) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / 0.07
    return logits, labels

def get_data(df):
    mols = [Chem.MolFromSmiles(x) for x in df['SMILES']]
    g = [smiles_to_bigraph(m, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer) for m in df['SMILES']]

    # 二分类标签转换
    # y = np.array(df['NEW-CONCENTRATION'] >= -6, dtype=np.int64)  # ≥ -6 为 1，< -6 为 0
    y = np.array(df['label'], dtype=np.float32)
    return g, y

def collate(sample):
    _, list_num, graphs, labels,pretrain_feats, index = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return _, list_num, batched_graph, torch.tensor(labels), torch.tensor(pretrain_feats), index

gcn_net = GCNPredictor(
        in_feats=74,
        hidden_feats=[300, 40],
        n_tasks=40,  # ✅ 回归任务设为1 60 20 40
        predictor_hidden_feats=10,
        predictor_dropout=0.5
    ).to(device)     # 你的GCN模型类
model_trans = Transformer_test().to(device)   # Transformer编码器
model_tgcn = Model_TGCN().to(device)       # 多模态融合主模型


checkpoint = torch.load('best_model_by_mcc.pth', map_location='cuda')  # 或 'cpu'

# 加载各子模块的 state_dict
gcn_net.load_state_dict(checkpoint['gcn_net'])
model_trans.load_state_dict(checkpoint['model_trans'])
model_tgcn.load_state_dict(checkpoint['model_tgcn'])

# 设置为 eval 模式（如果你用于推理或测试）
gcn_net.eval()
model_trans.eval()
model_tgcn.eval()
PATH_x_test = '/tmp/pycharm_project_GSToxi/TOXI_data2/protein_train1002.csv'
PATH_x_pretest = '/tmp/pycharm_project_GSToxi/TOXI_data2/protein_train1002_plus_prott5.npy'
x_pretest = np.load(PATH_x_pretest)

node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e')

#与 node_featurizer 相同，提取 原子级特征，但将数据存入 ndata['feat'] 字段，而不是 ndata['h']
atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
#使用 RDKit 解析 苯（benzene, c1ccccc1） 的分子结构，并获取 atom_featurizer（CanonicalAtomFeaturizer）的原子特征维度
mol = Chem.MolFromSmiles('c1ccccc1')
n_feats = atom_featurizer.feat_size('feat')

df_seq_test, y_test_tensor, y_true_test, list_num_test = func(PATH_x_test)

test_X = pd.read_csv(PATH_x_test)
x_test, y_test = get_data(test_X)
test_data = list(zip(df_seq_test, list_num_test, x_test, y_test,x_pretest, [i for i in range(len(test_X))]))
test_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate,
                                  drop_last=True)

gcn_list, trans_list, fusion_list, label_list,pre_list = [], [], [], [],[]
loss_infonce = torch.nn.CrossEntropyLoss().to(device)
epoch_loss = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for i, (X, list_num, graph, labels, pretrain_feats, index) in enumerate(test_loader_test):
        labels = labels.to(device).float()
        pretrain_feats = pretrain_feats.to(device).float()

        graph = graph.to(device)
        atom_feats = graph.ndata.pop('h').to(device)
        pred = gcn_net(graph, atom_feats)

        # 序列特征
        X = torch.cat(X, dim=0)  # List of [1, 64] → [batch, 64]
        X = torch.reshape(X, [batch_size, 50]).to(device)

        # 数值特征
        list_num = torch.tensor([item.detach().cpu().numpy() for item in list_num]).to(device)
        y, y_p = model_trans(X)

        hid_pairs = torch.cat([y_p, pred], 0)
        logits, cont_labels = info_nce_loss(hid_pairs)
        l_infonce = 0.01 * loss_infonce(logits, cont_labels)

        y, g_emb, t_emb, f_emb = model_tgcn(y, pred, list_num, pretrain_feats)

        y = y.to(device)
        y = torch.reshape(y, [batch_size])

        loss = nn.BCELoss()(y, labels.float()) + l_infonce
        epoch_loss += loss.item()

        pred_cls = y.detach().cpu().numpy()
        true_label = labels.to('cpu').numpy()

        all_preds.extend(pred_cls)
        all_labels.extend(true_label)

# 计算评估指标
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 将概率转为0/1
binary_preds = (all_preds >= 0.5).astype(int)

# 评估指标
SE = recall = np.sum((binary_preds == 1) & (all_labels == 1)) / (np.sum(all_labels == 1) + 1e-8)
AUROC = roc_auc_score(all_labels, all_preds)
AUPRC = average_precision_score(all_labels, all_preds)
F1 = f1_score(all_labels, binary_preds)
MCC = matthews_corrcoef(all_labels, binary_preds)

# 输出
print(f"Loss   : {epoch_loss:.4f}")
print(f"SE     : {SE:.4f}")
print(f"AUROC  : {AUROC:.4f}")
print(f"AUPRC  : {AUPRC:.4f}")
print(f"F1     : {F1:.4f}")
print(f"MCC    : {MCC:.4f}")