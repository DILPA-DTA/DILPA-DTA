from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from torch.utils.data import Dataset
from functools import partial
import MDAnalysis as mda
from scipy import sparse as sp
from scipy.spatial import distance_matrix


# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预定义的残基类型和金属元素集合
RESIDUE_TYPES = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                 'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                 'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                 'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']
METALS = {"LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI",
          "PT", "ZN", "CO", "PD", "CR", "FE", "V", "MN", "HG", "GA",
          "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA",
          "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND", "GD", "TB",
          "DY", "ER", "TM", "LU", "HF", "ZR", "U", "PU", "TH"}


# 功能函数：根据残基类型返回特征向量
def one_hot_encode_residue(res_name):
    if res_name in METALS:
        res_name = "M"  # 将金属离子归为"M"
    encoding = [0] * len(RESIDUE_TYPES)
    if res_name in RESIDUE_TYPES:
        encoding[RESIDUE_TYPES.index(res_name)] = 1
    return encoding


# 功能函数：计算残基之间的距离
def calculate_distance_matrix(residues):
    positions = [res["CA_position"] for res in residues]
    return distance_matrix(positions, positions)


# 功能函数：提取残基CA原子的坐标
def get_ca_position(res):
    if res.resname in METALS:
        return res.atoms.positions[0]
    ca_atoms = res.atoms.select_atoms("name CA")
    return ca_atoms.positions[0] if len(ca_atoms) > 0 else res.atoms.positions.mean(axis=0)


# 主函数：将蛋白质PDB文件转换为图




def prot_to_graph(pdb_id, pdb_path, cutoff=10.0, pos_enc_dim=8,device="cuda:3"):

    u = mda.Universe(pdb_path)
    residues = [{"resname": res.resname, "CA_position": get_ca_position(res)} for res in u.residues]

    # 创建DGL图，节点数等于残基数
    num_residues = len(residues)
    g = dgl.graph(([], []), num_nodes=num_residues,device=device)

    # 添加节点特征：对残基类型进行one-hot编码
    res_feats = np.array([one_hot_encode_residue(res["resname"]) for res in residues])
    g.ndata["feats"] = torch.tensor(res_feats, dtype=torch.float32).to(device)

    # 计算残基间距离矩阵
    distance_mat = calculate_distance_matrix(residues)
    edge_list = []
    edge_dists = []

    # 根据cutoff值添加边
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            if distance_mat[i, j] <= cutoff:
                edge_list.append((i, j))
                edge_dists.append(distance_mat[i, j] * 0.1)  # 转换单位

    # 创建双向边，并为边添加距离特征
    src, dst = zip(*edge_list)
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    g.edata["dist"] = torch.tensor(edge_dists + edge_dists, dtype=torch.float32).to(device)

    # 添加拉普拉斯位置编码
    g = laplacian_positional_encoding(g, pos_enc_dim)
    if g is None or g.num_nodes() == 0:
        return None

    return g





# 拉普拉斯位置编码函数

import numpy as np
import scipy.sparse as sp

import dgl
import numpy as np
import scipy.sparse as sp
import torch


def laplacian_positional_encoding(g, pos_enc_dim):
    # 获取邻接矩阵并转换为 SciPy CSR 格式
    A = g.adjacency_matrix().to_dense().cpu().numpy()  # 转换为密集 numpy 矩阵
    A = sp.csr_matrix(A)  # 转换为 SciPy 稀疏矩阵格式

    # 计算归一化拉普拉斯矩阵 L = I - D^-1/2 * A * D^-1/2
    degrees = g.in_degrees().cpu().numpy()  # 将度数从 GPU 移动到 CPU
    N = sp.diags(np.power(np.clip(degrees, 1, None), -0.5))  # 归一化的度矩阵
    L = sp.eye(g.number_of_nodes()) - N @ A @ N  # 拉普拉斯矩阵

    # 计算特征值和特征向量
    EigVal, EigVec = np.linalg.eigh(L.toarray())
    idx = EigVal.argsort()  # 对特征值排序
    EigVec = np.real(EigVec[:, idx])

    # 若特征向量数少于需要的维度，则进行填充
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), axis=1)

    # 将拉普拉斯位置编码添加为图的节点特征
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().to(g.device)

    return g



import itertools
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem



class DTIDataset(Dataset):

    def __init__(self, list_IDs, df,encoder=ProteinPocketEncoder(input_dim=32, hidden_dim=62, num_layers=3, device='cuda'), max_drug_nodes=600, cutoff=10.0, pdb_dir="/home/qwe/data/lijianguang/bibranch/BINDTI/datasets/pdb_files"):
        self.list_IDs = list_IDs
        self.df = df
        self.encoder = encoder
        self.max_drug_nodes = max_drug_nodes
        self.cutoff = cutoff
        self.pdb_dir = pdb_dir
        # Define atom and bond featurizers for drug molecules
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        v_d.smiles = self.df.iloc[index]['SMILES']
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74),
                                       torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {'h': virtual_node_feat})
        v_d = v_d.add_self_loop()

        #v_d = integrate_3d_info(smiles=smiles,v_d=v_d, cutoff=3.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        #Process protein 3D structure from PDB file
        pdb_id = self.df.iloc[index]['PDBID']
        pdb_path = f"{self.pdb_dir}/{pdb_id}.pdb"
        device = torch.device("cuda:3")
        v_p = torch.tensor(integer_label_protein(self.df.iloc[index]['Protein'])).to(device)
        pocket_graph = prot_to_graph(pdb_id, pdb_path, cutoff=self.cutoff,device="cuda:3")
        # print("p 的类型:", type(p), "xingzhuang:", p.size())
        v_p.cpu()

        # Get interaction label
        y = self.df.iloc[index]['Y']

        return v_d, v_p, y, pocket_graph
