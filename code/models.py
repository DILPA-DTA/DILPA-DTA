import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from DCA import DCA
from Bi_Interaction import BiInteraction
import torchvision.ops  as ops
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import dgl.nn.pytorch as dglnn
import scipy.stats
import numpy as np
def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)

    return n, loss

# def binary_cross_entropy(pred_output, labels, alpha=0.8, gamma=2.5):
#     # n = pred_output.view(-1, 1)  # 确保形状正确
#     m = nn.Sigmoid()
#     n = torch.squeeze(m(pred_output), 1)
#     loss = ops.sigmoid_focal_loss(n, labels, alpha=alpha, gamma=gamma, reduction='mean')
#     return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.sze(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss



class DILPADTA(nn.Module):
    def __init__(self, device='cuda', dropout_rate=0.3,**config):
        super(DILPADTA, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']


        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinDCA(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        self.cross_intention = BiInteraction(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer,
                                           device=device)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary,dropout_rate=dropout_rate)


    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)  # v_d.shape(64, 290, 128)
        v_p = self.protein_extractor(v_p)  # v_p.shape:(64, 1200, 128)
        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)  # f:[64, 256]
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att



class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=F.relu):
        """
        :param in_feats: 输入节点特征维度
        :param dim_embedding: 初始全连接层输出的特征维度
        :param padding: 是否对最后一行参数进行填0处理
        :param hidden_feats: 一个列表，每个元素为GIN层的输出维度
        :param activation: 激活函数，默认使用ReLU
        """
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)

        self.activation = activation

        # 构造多个GIN层，每一层均使用MLP作为聚合函数
        self.gnn_layers = nn.ModuleList()
        last_dim = dim_embedding
        for hidden_dim in hidden_feats:
            # 定义两层MLP
            mlp = nn.Sequential(
                nn.Linear(last_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # GINConv使用'sum'作为聚合方式
            gin_conv = dglnn.GINConv(mlp, 'sum')
            self.gnn_layers.append(gin_conv)
            last_dim = hidden_dim
        self.output_feats = last_dim

    def forward(self, batch_graph):
        # 获取节点特征，DGL图中节点特征通常存储在ndata['h']
        node_feats = batch_graph.ndata.pop('h')
        # 初始特征变换
        node_feats = self.init_transform(node_feats)
        # 逐层GIN卷积
        for gin_layer in self.gnn_layers:
            residual = node_feats  # 残差连接
            node_feats = gin_layer(batch_graph, node_feats)
            if self.activation is not None:
                node_feats = self.activation(node_feats)
            node_feats = node_feats + residual  # 添加残差
        # 将节点特征按batch_size进行reshape，适应后续处理
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinDCA(nn.Module):
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(ProteinDCA, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]

        self.acmix1 = ACmix(in_planes=in_ch[0], out_planes=in_ch[1], head=num_head)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.acmix2 = ACmix(in_planes=in_ch[1], out_planes=in_ch[2], head=num_head)
        self.bn2 = nn.BatchNorm1d(in_ch[2])

        self.acmix3 = ACmix(in_planes=in_ch[2], out_planes=in_ch[3], head=num_head)
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)  # 64*128*1200

        v = self.bn1(F.relu(self.acmix1(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn2(F.relu(self.acmix2(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn3(F.relu(self.acmix3(v.unsqueeze(-2))).squeeze(-2))
        v = v.view(v.size(0), v.size(2), -1)
        return v



class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, dropout_rate=0.5, num_heads=8):
        """
        :param in_dim: 输入特征维度
        :param hidden_dim: MLP 中间层维度
        :param out_dim: MLP最后一层输出的特征维度，用于趋势提取
        :param binary: 最终输出维度（例如1表示回归单值）
        :param dropout_rate: Dropout 概率
        :param num_heads: 多头注意力中头的数量
        """
        super(MLPDecoder, self).__init__()
        # MLP主分支
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)


        self.fc4 = nn.Linear(out_dim, binary)


        self.trend_query = nn.Parameter(torch.randn(1, 1, out_dim))
        self.mha = nn.MultiheadAttention(embed_dim=out_dim, num_heads=num_heads, batch_first=True)
        self.linear_trend = nn.Linear(out_dim, binary)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # x: [B, in_dim]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))

        output_mlp = self.fc4(x)


        x_attn = x.unsqueeze(1)
        trend_query = self.trend_query.expand(x.size(0), -1, -1)
        attn_output, _ = self.mha(trend_query, x_attn, x_attn)
        output_trend = self.linear_trend(attn_output.squeeze(1))

        return output_mlp + output_trend
