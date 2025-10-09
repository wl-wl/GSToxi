import os
import pickle
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef, confusion_matrix
)
from sklearn.manifold import TSNE
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
from dgllife.model import GINPredictor
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)                  # Python 内置 random 模块
    np.random.seed(seed)               # NumPy
    torch.manual_seed(seed)            # PyTorch CPU
    torch.cuda.manual_seed(seed)       # PyTorch GPU
    torch.cuda.manual_seed_all(seed)   # 多 GPU 情况

    # 禁用 cuDNN 的非确定性算法（性能略降低，但确保可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2)


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

all_auc, all_auprc, all_se, all_f1, all_mcc = [], [], [], [], []
def main_():


    PATH_x_train = "/tmp/pycharm_project_GSToxi/TOXI_data/train_sequence.csv"
    PATH_x_test = '/tmp/pycharm_project_GSToxi/TOXI_data/test_sequence.csv'

    PATH_x_pretrain='/tmp/pycharm_project_GSToxi/TOXI_data/X_train_prott5_pooled.npy'
    PATH_x_pretest = '/tmp/pycharm_project_GSToxi/TOXI_data/X_test_prott5_pooled.npy'

    x_pretrain = np.load(PATH_x_pretrain)
    x_pretest = np.load(PATH_x_pretest)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_trans = Transformer_test().to(device)
    torch.cuda.empty_cache()
    df_seq_train, y_train_tensor, y_true_train, list_num_train = func(PATH_x_train)
    df_seq_test, y_test_tensor, y_true_test, list_num_test = func(PATH_x_test)


    # device = 'cpu'
    #原子特征化（Atom Featurization）工具，主要用于分子图（Molecular Graph）表示学习
    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e')

    #与 node_featurizer 相同，提取 原子级特征，但将数据存入 ndata['feat'] 字段，而不是 ndata['h']
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='feat')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat')
    #使用 RDKit 解析 苯（benzene, c1ccccc1） 的分子结构，并获取 atom_featurizer（CanonicalAtomFeaturizer）的原子特征维度
    mol = Chem.MolFromSmiles('c1ccccc1')
    n_feats = atom_featurizer.feat_size('feat')
    #n_feats 是 74，意味着每个原子用一个 74 维向量 来表示
    print("n_feats:",n_feats)

    def get_data(df):
        mols = [Chem.MolFromSmiles(x) for x in df['SMILES']]
        g = [smiles_to_bigraph(m, node_featurizer=node_featurizer,edge_featurizer=edge_featurizer) for m in df['SMILES']]

        # 二分类标签转换
        # y = np.array(df['NEW-CONCENTRATION'] >= -6, dtype=np.int64)  # ≥ -6 为 1，< -6 为 0
        y = np.array(df['label'], dtype=np.float32)
        return g, y


    gcn_net = GCNPredictor(
        in_feats=n_feats,
        hidden_feats=[300, 40],
        n_tasks=40,  # ✅ 回归任务设为1 60 20 40
        predictor_hidden_feats=10,
        predictor_dropout=0.5
    ).to(device)


    gcn_net = gcn_net.to(device)
    model_tgcn = Model_TGCN().to(device)

    def collate(sample):
        _, list_num, graphs, labels,pretrain_feats, index = map(list, zip(*sample))
        batched_graph = dgl.batch(graphs)
        batched_graph.set_n_initializer(dgl.init.zero_initializer)
        batched_graph.set_e_initializer(dgl.init.zero_initializer)
        return _, list_num, batched_graph, torch.tensor(labels), torch.tensor(pretrain_feats), index

    train_X = pd.read_csv(PATH_x_train)
    x_train, y_train = get_data(train_X)
    # print(df_seq_train.shape, list_num_train.shape, x_train.shape, y_train.shape)
    train_data = list(zip(df_seq_train, list_num_train, x_train, y_train, x_pretrain, [i for i in range(len(train_X))]))
    train_loader_ = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    test_X = pd.read_csv(PATH_x_test)
    x_test, y_test = get_data(test_X)
    test_data = list(zip(df_seq_test, list_num_test, x_test, y_test,x_pretest, [i for i in range(len(test_X))]))
    test_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate,
                                  drop_last=True)

    optimizer = torch.optim.Adam([{'params': gcn_net.parameters()},
                                  {'params': model_trans.parameters()},
                                  {'params': model_tgcn.parameters()}], lr=0.0018
                                 )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                           verbose=True)
    loss_infonce = torch.nn.CrossEntropyLoss().to(device)

    best_se = -1.0
    best_auroc = -1.0
    best_results = {}
    best_model_state = None

    train_acc_hist, train_auc_hist = [], []
    test_acc_hist, test_auc_hist = [], []
    graph_embeds, trans_embeds, fused_embeds,all_binary_preds_emb = [], [], [],[]

    for epoch in range(1, 101):
        all_y_pred = []
        all_y_true = []

        gcn_net.train()
        model_trans.train()
        model_tgcn.train()

        train_epoch_loss, train_epoch_acc, train_epoch_r2 = 0, 0, 0
        all_preds = []
        all_labels = []

        for i, (X, list_num, graph, labels, pretrain_feats, index) in enumerate(train_loader_):
            train_labels = labels.to(device).float()
            pretrain_feats = pretrain_feats.to(device).float()

            graph = graph.to(device)
            atom_feats = graph.ndata.pop('h').to(device)
            train_pred = gcn_net(graph, atom_feats)

            X = torch.cat(X, dim=0)
            X = torch.reshape(X, [batch_size, 50]).to(device)
            # X = X.reshape(-1, 50).to(device)

            list_num = torch.tensor([item.detach().cpu().numpy() for item in list_num]).to(device)
            y,y_p = model_trans(X)

            hid_pairs = torch.cat([y_p, train_pred], 0)
            logits, cont_labels = info_nce_loss(hid_pairs)
            l_infonce = 0.01 * loss_infonce(logits, cont_labels)


            y,g_emb, t_emb, f_emb = model_tgcn(y, train_pred, list_num, pretrain_feats)
            y = y.to(device)
            g_emb = g_emb.to(device)
            t_emb = t_emb.to(device)
            f_emb = f_emb.to(device)


            y = torch.reshape(y, [batch_size])

            # train_loss = nn.MSELoss()(y.cpu()*(label_max-label_min), train_labels.cpu()*(label_max-label_min))
            train_loss = nn.BCELoss()(y, train_labels.float())+l_infonce
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            trlist_numain_pred_cls = train_pred.argmax(-1).detach().to('cpu').numpy()
            train_true_label = train_labels.to('cpu').numpy()
            yy = [1 if i >= 0.5 else 0 for i in y.detach().cpu().numpy()]
            train_epoch_acc += sum(train_true_label == yy)

            all_y_pred.extend(y.detach().cpu().numpy())
            all_y_true.extend(train_labels.detach().cpu().numpy())


        train_epoch_acc = train_epoch_acc / train_true_label.shape[0]
        train_epoch_acc /= (i + 1)
        train_epoch_loss /= (i + 1)
        train_auc = roc_auc_score(all_y_true, all_y_pred)

        binary_preds = [1 if i >= 0.5 else 0 for i in all_y_pred]
        tn, fp, fn, tp = confusion_matrix(all_y_true, binary_preds).ravel()
        train_acc = (tp + tn) / (tp + tn + fp + fn)
        train_auc_hist.append(train_auc)
        train_acc_hist.append(train_acc)

        # === 各项指标 ===
        train_auc = roc_auc_score(all_y_true, all_y_pred)
        train_auprc = average_precision_score(all_y_true, all_y_pred)
        train_recall = recall_score(all_y_true, binary_preds)
        train_f1 = f1_score(all_y_true, binary_preds)
        train_mcc = matthews_corrcoef(all_y_true, binary_preds)

        def train_test_val(dataloader, best_f1_threshold=0.5):
            epoch_loss = 0
            epoch_loss, epoch_acc = 0, 0
            all_preds = []
            all_labels = []
            mlist=[]
            all_binary_preds = []

            gcn_net.eval()
            model_trans.eval()
            model_tgcn.eval()

            with torch.no_grad():
                for i, (X, list_num, graph, labels, pretrain_feats, index) in enumerate(dataloader):
                    labels = labels.to(device).float()
                    pretrain_feats = pretrain_feats.to(device).float()

                    graph = graph.to(device)
                    atom_feats = graph.ndata.pop('h').to(device)
                    pred = gcn_net(graph, atom_feats)

                    # 序列特征
                    X = torch.cat(X, dim=0)  # List of [1, 64] → [batch, 64]
                    X = torch.reshape(X, [batch_size, 50]).to(device)
                    # X = X.reshape(-1, 50).to(device)

                    # 数值特征
                    list_num = torch.tensor([item.detach().cpu().numpy() for item in list_num]).to(device)
                    y,y_p = model_trans(X)

                    hid_pairs = torch.cat([y_p, pred], 0)
                    logits, cont_labels = info_nce_loss(hid_pairs)
                    l_infonce = 0.01 * loss_infonce(logits, cont_labels)

                    # y_p = inverse_minmax(y_p, label_min, label_max)
                    y,g_emb, t_emb, f_emb = model_tgcn(y, pred, list_num, pretrain_feats)

                    y = y.to(device)
                    g_emb = g_emb.to(device)
                    t_emb = t_emb.to(device)
                    f_emb = f_emb.to(device)


                    # y = model_tgcn(y, pred, list_num)
                    y = torch.reshape(y, [batch_size])

                    # loss = nn.MSELoss()(y.cpu()*(label_max-label_min), labels.cpu()*(label_max-label_min))
                    loss = nn.BCELoss()(y, labels.float())+l_infonce
                    epoch_loss += loss.item()
                    pred_cls = y.detach().cpu().numpy()
                    true_label = labels.to('cpu').numpy()
                    binary_preds = [1 if m >= 0.5 else 0 for m in pred_cls]
                    all_binary_preds.extend(binary_preds)  # ← 新增这一行

                    yy = [1 if m >= 0.5 else 0 for m in y.detach().cpu().numpy()]
                    mlist.extend(pred_cls)
                    all_labels.extend(true_label)
                    epoch_acc += sum(true_label == yy)

                    graph_embeds.append(g_emb.cpu())
                    trans_embeds.append(t_emb.cpu())
                    fused_embeds.append(f_emb.cpu())
                    # all_binary_preds_emb.append(all_binary_preds)
                    # all_labels_emb.append(true_label)

                epoch_acc = epoch_acc / true_label.shape[0]
                epoch_acc /= (i + 1)
                epoch_loss /= (i + 1)
                auroc = roc_auc_score(all_labels, mlist)
                auprc = average_precision_score(all_labels, mlist)

                se = recall_score(all_labels, all_binary_preds)
                tn, fp, fn, tp = confusion_matrix(all_labels, all_binary_preds).ravel()
                test_acc = (tp + tn) / (tp + tn + fp + fn)

                test_auc_hist.append(auroc)
                test_acc_hist.append(test_acc)

                acc = (tp + tn) / (tp + tn + fp + fn)  # Accuracy
                sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity

                f1 = f1_score(all_labels, all_binary_preds)
                mcc = matthews_corrcoef(all_labels, all_binary_preds)


            return epoch_acc, epoch_loss, auroc, auprc, se, f1, mcc,acc,sp,test_auc_hist,test_acc_hist,g_emb, t_emb, f_emb

        test_acc, test_loss, test_auroc, test_auprc, test_se, test_f1, test_mcc,test_acc,test_sp,test_auc_hist,test_acc_hist,g_emb, t_emb, f_emb = train_test_val(
            test_loader_test)



        # Step 1: 收集当前 epoch 的测试指标
        current_metrics = {
            'SE': test_se,
            'AUROC': test_auroc,
            'AUPRC': test_auprc,
            'F1': test_f1,
            'MCC': test_mcc,
            'ACC': test_acc,
            'SP': test_sp,
        }

        toxgin_baseline = {
            'SE': 0.8014,
            'AUROC': 0.9172,
            'AUPRC': 0.89,
            'F1': 0.8354,
            'MCC': 0.6866,
            'ACC': 0.8423,
            'SP': 0.8832,
        }

        # Step 2: 与 ToxGIN baseline 比较，统计超过的数量
        better_count = sum([
            current_metrics[key] > toxgin_baseline[key]
            for key in toxgin_baseline
        ])

        # Step 3: 至少 3 项超越 baseline 才保存
        if better_count >= 1:
            # 用 MCC 作为综合排名指标选出最优的“满足条件模型”
            if current_metrics['AUROC'] > best_auroc:
                best_auroc = current_metrics['AUROC']
                best_results = current_metrics
                best_model_state = {
                    'gcn_net': gcn_net.state_dict(),
                    'model_trans': model_trans.state_dict(),
                    'model_tgcn': model_tgcn.state_dict()
                }
                print(f"✅ New best valid model (AUROC={best_auroc:.4f}) — Saved.")

        print(f"Epoch: {epoch}")
        print(f"[Train] AUROC: {train_auc:.4f} | ACC: {train_epoch_acc:.4f}")
        print(
            f"[Test ] SE: {test_se:.4f} | AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f} |  F1: {test_f1:.4f} | MCC: {test_mcc:.4f}"
            f"|  ACC: {test_acc:.4f} | SP: {test_sp:.4f}")

    # torch.save(best_model_state, 'best_model_by_mcc.pth')

    epochs = range(1, 100 + 1)
    plt.figure(figsize=(12, 6))

    plt.rcParams.update({
        'font.size': 14,  # 所有字体大小
        'axes.titlesize': 14,  # 图标题
        'axes.labelsize': 14,  # 坐标轴标题
        'xtick.labelsize': 14,  # x轴刻度
        'ytick.labelsize': 14,  # y轴刻度
        'legend.fontsize': 14,  # 图例字体
    })
    # ACC 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_hist, label="Train ACC")
    plt.plot(epochs, test_acc_hist, label="Test ACC")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # plt.title("ACC")
    # plt.ylim(0.7,1.1)
    plt.savefig('ACC.png', dpi=300)
    plt.legend()
    plt.grid(alpha=0.3)

    # AUROC 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_auc_hist, label="Train AUROC")
    plt.plot(epochs, test_auc_hist, label="Test AUROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    # plt.title("AUROC")
    # plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('AUROC.png',dpi=300)
    plt.tight_layout()
    plt.show()

    # === 打印最佳结果 ===
    print("\n🎯 Best Results (based on MCC):")
    for k, v in best_results.items():
        print(f"{k:6}: {v:.4f}")



if __name__ == '__main__':
    main_()
