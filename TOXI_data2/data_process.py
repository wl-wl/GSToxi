import pandas as pd
from rdkit import Chem

# 读取数据
df = pd.read_csv("/tmp/pycharm_project_GSToxi/TOXI_data2/protein_train1002.csv")

# 定义SMILES转换函数
def seq_to_smiles(seq):
    mol = Chem.MolFromFASTA(seq)
    return Chem.MolToSmiles(mol) if mol else "Invalid"

# 定义AAC计算函数
def compute_aac(seq):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq_len = len(seq)
    aac = {}
    for aa in amino_acids:
        aac[f"AAC_{aa}"] = seq.upper().count(aa) / seq_len if seq_len > 0 else 0.0
    return aac

# 生成新列
df["SMILES"] = df["sequence"].apply(seq_to_smiles)
aac_features = df["sequence"].apply(compute_aac).apply(pd.Series)

# 合并结果
df_final = pd.concat([df, aac_features], axis=1)

# 保存到新文件
df_final.to_csv("/tmp/pycharm_project_GSToxi/TOXI_data2/protein_train1002_with_SMILES_AAC.csv", index=False)
