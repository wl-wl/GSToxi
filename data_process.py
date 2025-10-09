import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from torch.nn.utils.rnn import pad_sequence
from ast import literal_eval


warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_dataset_number(PATH_x):
    df = pd.read_csv(PATH_x)
    df_num = df.drop(columns=[
          # 唯一 ID
        "sequence",  # 数据来源
        "SMILES",  # 发表年份
        "label"
    ]).values
    print("Dataset number: ", len(df_num[0]))
    dfy = pd.read_csv(PATH_x)
    y = dfy['label'].values.reshape(-1,1)
    y = y.astype('float32')
    df_num = scale(df_num)
    print("df_num",df_num.shape)
    return df_num,y



def create_dataset_list(PATH_x):
    # 20个氨基酸单字母代码
    aa_list = [
        "A", "C", "D", "E", "F",
        "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R",
        "S", "T", "V", "W", "Y"
    ]
    # 读取CSV中这20列
    df_list = pd.read_csv(PATH_x, usecols=aa_list)
    # 将字符串形式的列表解析为真正的列表（使用 safer 的 literal_eval）
    for aa in aa_list:
        df_list[aa] = df_list[aa].apply(lambda x: literal_eval(x))
    # 找出最长序列长度（假设所有AA列的长度一致，按任意一列即可）
    max_len = max(len(x) for x in df_list[aa_list[0]])
    print('最长序列max_len', max_len)
    # 初始化最终拼接数据
    all_features = []
    for aa in aa_list:
        aa_values = df_list[aa].values
        data_padded = np.zeros((len(aa_values), max_len))
        for i, row in enumerate(aa_values):
            data_padded[i, :len(row)] = row
        all_features.append(data_padded)
    # 在特征维度拼接，最终 shape = [N, 20 * max_len]
    full_data = np.concatenate(all_features, axis=1)
    # 标准化
    full_data = scale(full_data)
    # 转为 tensor
    tensor_data = torch.tensor(full_data, dtype=torch.float32)
    return tensor_data


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

# def create_dataset_seq(PATH_x):
#     """
#     :param PATH_x:
#     :return:
#
#     """
#     df = pd.read_csv(PATH_x,usecols=['SMILES'])
#     vocab = []
#     datas = []
#
#     for i, row in df.iterrows():
#         data = row["SMILES"]
#
#         tokens = smi_tokenizer(data).split(" ")
#         if len(tokens) <= 256:
#             di = tokens+["PAD"]*(256-len(tokens))
#         else:
#             di = tokens[:256]
#         datas.append(di)
#         vocab.extend(tokens)
#     vocab = list(set(vocab))
#     vocab = ["PAD"]+vocab
#     with open("vocab.txt","w",encoding="utf8") as f:
#         for i in vocab:
#             f.write(i)
#             f.write("\n")
#     mlist = []
#     word2id = {}
#     for i,d in enumerate(vocab):
#         word2id[d] = i
#     for d_i in datas:
#         mi = [word2id[d] for d in d_i]
#         mlist.append(np.array(mi))
#
#     return mlist


def create_dataset_seq(PATH_x):
    """
    读取PATH_x中的SEQUENCE列，对氨基酸序列进行tokenize与编码，固定长度为60
    """
    df = pd.read_csv(PATH_x, usecols=['sequence'])
    vocab = set()
    datas = []

    max_len = 50

    for _, row in df.iterrows():
        sequence = row["sequence"]
        tokens = list(sequence)  # 逐字符作为氨基酸残基进行token化

        if len(tokens) <= max_len:
            tokens_padded = tokens + ["PAD"] * (max_len - len(tokens))
        else:
            tokens_padded = tokens[:max_len]
            # print("有无拼接？！")

        datas.append(tokens_padded)
        vocab.update(tokens)

    vocab = ["PAD"] + sorted(vocab)  # 保证PAD是id=0，其他按字典序排列

    # 写入vocab文件
    with open("vocab.txt", "w", encoding="utf8") as f:
        for token in vocab:
            f.write(token + "\n")

    # 构建word2id字典
    word2id = {token: idx for idx, token in enumerate(vocab)}

    # 将每条序列转换为ID序列
    encoded_list = []
    for tokens in datas:
        encoded = [word2id[token] for token in tokens]
        encoded_list.append(np.array(encoded))

    return encoded_list




def func(PATH):
    df_num, y_true = create_dataset_number(PATH)
    df_seq = create_dataset_seq(PATH)
    df_seq = torch.tensor([item for item in df_seq]).to(torch.int64)
    tensor_data_num = torch.tensor(df_num, dtype=torch.float32)
    y = torch.tensor([item for item in y_true]).to(torch.float)
    return df_seq, y, y_true,tensor_data_num


