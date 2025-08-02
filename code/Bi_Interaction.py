import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import seaborn as sns
class Intention(nn.Module):
    def __init__(self, dim, num_heads, kqv_bias=False, device='cuda'):
        super(Intention, self).__init__()
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        self.device = device

        assert dim % num_heads == 0, 'dim must be divisible by num_heads!'

        self.wq = nn.Linear(dim, dim, bias=kqv_bias)
        self.wk = nn.Linear(dim, dim, bias=kqv_bias)
        self.wv = nn.Linear(dim, dim, bias=kqv_bias)

        self.out = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-2)

        self.alpha = nn.Parameter(torch.rand(1))  # original parameter
        self.temperature = nn.Parameter(torch.tensor(1.0))  # added temperature parameter
        self.norm = nn.LayerNorm(dim)  # LayerNorm for stability

    def forward(self, x, query=None):
        residual = x  # For residual connection

        if query is None:
            query = x

        query = self.wq(query)
        key = self.wk(x)
        value = self.wv(x)

        b, n, c = x.shape
        key = key.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)
        key_t = key.clone().permute(0, 1, 3, 2)
        value = value.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        b, n, c = query.shape
        query = query.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        # Stable inverse with regularization
        kk = key_t @ key  # (B, H, D, D)
        kk = self.alpha * torch.eye(kk.shape[-1], device=self.device) + kk
        kk_inv = torch.inverse(kk)

        # Attention
        attn_map = (kk_inv @ key_t) @ value
        attn_map = self.softmax(attn_map / self.temperature)

        out = query @ attn_map
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out = self.out(out)

        # Residual + LayerNorm
        if out.shape == residual.shape:
            out = self.norm(out + residual)
        else:
            out = self.norm(out)

        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wk = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wv = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        att, _ = self.attn(query, key, value)
        out = att + x
        return out


class IntentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, kqv_bias=True, device='cuda'):
        super(IntentionBlock, self).__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.attn = Intention(dim=dim, num_heads=num_heads, kqv_bias=kqv_bias, device=device)
        self.softmax = nn.Softmax(dim=-2)
        self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x, q):
        x = self.norm_layer(x)
        q_t = q.permute(0, 2, 1)
        att = self.attn(x, q)
        att_map = self.softmax(att)
        out = self.beta * q_t @ att_map
        return out


class BiInteraction(nn.Module):
    def __init__(self, embed_dim, layer=1, num_head=8, device='cuda'):
        super(BiInteraction, self).__init__()

        self.layer = layer
        self.drug_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        self.protein_intention = nn.ModuleList([
            IntentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])

        # Self-attention layers for drug and protein features
        self.attn_drug = SelfAttention(dim=embed_dim, num_heads=num_head)
        self.attn_protein = SelfAttention(dim=embed_dim, num_heads=num_head)

        # Cross-attention mechanism for compound-protein interaction
        self.compound_attention = nn.Linear(embed_dim, embed_dim)
        self.protein_attention = nn.Linear(embed_dim, embed_dim)
        self.inter_attention = nn.Linear(embed_dim, embed_dim)

    def forward(self, drug, protein):
        drug = self.attn_drug(drug)
        protein = self.attn_protein(protein)

        # Multi-level interaction between drug and protein
        for i in range(self.layer):
            v_p = self.drug_intention[i](drug, protein)
            v_d = self.protein_intention[i](protein, drug)
            drug, protein = v_d, v_p

        # Cross-attention to capture interaction features
        src_att = self.protein_attention(protein)
        trg_att = self.compound_attention(drug)
        c_att = trg_att.unsqueeze(2).repeat(1, 1, protein.shape[1], 1)
        p_att = src_att.unsqueeze(1).repeat(1, drug.shape[1], 1, 1)
        attention_matrix = self.inter_attention(torch.relu(c_att + p_att)).squeeze(-1)
        # print(attention_matrix.shape)
        attn_map = attention_matrix[0][0]  # shape: [L_d, L_p]
        attn_map1 = attn_map.reshape(64, 2, 64, 2).mean(dim=(1, 3)).detach().cpu().numpy()
        i = random.randint(0, 60)
        # print(i)
        plt.figure(figsize=(10, 6))
        sns.heatmap(attn_map1, cmap='viridis')
        plt.title("Attention Score")
        plt.xlabel("Protein Residue position")
        plt.ylabel("Drug Atom position")
        plt.tight_layout()
        plt.savefig(f"/home/qwe/data/lijianguang/bibranch/BINDTI/code/attention/attention_{i}.png", dpi=300)
        plt.show()
        # Mean pooling of attention for compound and protein
        compound_attention = torch.mean(attention_matrix, 2)
        protein_attention = torch.mean(attention_matrix, 1)

        # Apply attention pooling to drug and protein embeddings
        v_d1 = reduce(compound_attention.permute(0, 2, 1) * drug, 'B H W -> B H', 'max')
        v_p1 = reduce(protein_attention.permute(0, 2, 1) * protein, 'B H W -> B H', 'max')
        # i = random.randint(0, 60)

        v_d = reduce(drug, 'B H W -> B H', 'max')
        v_p = reduce(protein, 'B H W -> B H', 'max')
        f = torch.cat((v_d, v_p), dim=1)

        return f, v_d, v_p, None
