import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2, use_residual=True):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.use_residual = use_residual
        # 残差投影（如果输入输出维度不同）
        if use_residual and nfeat != nclass:
            self.res_proj = nn.Linear(nfeat, nclass)
        else:
            self.res_proj = None
        self.ln = nn.LayerNorm(nclass)

    def forward(self, x, adj):
        identity = x
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adj))
        # 残差连接
        if self.use_residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            x2 = x2 + identity
        x2 = self.ln(x2)
        return x2

# ================= 图注意力层 (GAT) =================
class GraphAttentionLayer(nn.Module):
    """
    图注意力层，支持有向图
    基于 "Graph Attention Networks" (Velickovic et al., ICLR 2018)
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha  # LeakyReLU的负斜率
        self.concat = concat
        
        # 特征变换矩阵
        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        # 注意力机制参数
        self.a = Parameter(torch.FloatTensor(2 * out_features, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / (self.W.size(1) ** 0.5)
        self.W.data.uniform_(-stdv, stdv)
        stdv = 1. / (self.a.size(0) ** 0.5)
        self.a.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        """
        Args:
            input: [N, in_features] 节点特征矩阵
            adj: 稀疏邻接矩阵 [N, N] (支持有向图)
        Returns:
            output: [N, out_features] 输出特征
        """
        # 特征变换: h' = hW
        h = torch.mm(input, self.W)  # [N, out_features]
        N = h.size(0)
        
        # 计算注意力分数（优化版本，只对存在的边计算）
        # Wh_i 和 Wh_j 分别表示源节点和目标节点的变换特征
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        
        # 如果邻接矩阵是稀疏的，只对存在的边计算注意力
        if adj.is_sparse:
            # 获取边的索引
            adj_coo = adj.coalesce()
            edge_index = adj_coo.indices()  # [2, E]
            row, col = edge_index[0], edge_index[1]
            
            # 计算边的注意力分数
            h_row = h[row]  # [E, out_features]
            h_col = h[col]  # [E, out_features]
            edge_features = torch.cat([h_row, h_col], dim=1)  # [E, 2*out_features]
            e = F.leaky_relu(torch.mm(edge_features, self.a).squeeze(), negative_slope=self.alpha)  # [E]
            
            # 【内存优化】使用分组操作对每行的边分别进行softmax，避免构建密集矩阵
            # 对每一行（源节点）的边进行softmax归一化
            
            # 方法：使用分组操作实现稀疏softmax（完全避免密集矩阵和原地操作）
            # 使用更向量化的方法，避免循环和原地操作
            
            # 1. 找到每行的最大值（用于数值稳定的softmax）
            # 使用分组操作：对每个源节点，找到其所有边的最大值
            unique_rows, inverse_indices = torch.unique(row, return_inverse=True)
            num_unique = len(unique_rows)
            
            # 计算每个组的最大值（非原地操作）
            row_max_grouped = torch.zeros(num_unique, device=input.device, dtype=input.dtype)
            for idx in range(num_unique):
                mask = (inverse_indices == idx)
                if mask.any():
                    row_max_grouped[idx] = e[mask].max()
            
            # 将分组结果映射回每条边
            row_max_per_edge = row_max_grouped[inverse_indices]  # [E]
            
            # 2. 计算 exp(e - max)，只对存在的边
            e_exp = torch.exp(e - row_max_per_edge)  # [E]
            
            # 3. 计算每行的归一化因子（sum of exp）
            # 使用分组操作计算每行的和
            row_sum_grouped = torch.zeros(num_unique, device=input.device, dtype=input.dtype)
            for idx in range(num_unique):
                mask = (inverse_indices == idx)
                if mask.any():
                    row_sum_grouped[idx] = e_exp[mask].sum()
            row_sum_grouped = row_sum_grouped.clamp(min=1e-8)  # 避免除零
            
            # 将分组结果映射回每条边
            row_sum_per_edge = row_sum_grouped[inverse_indices]  # [E]
            
            # 4. 归一化得到注意力权重
            attention_weights = e_exp / row_sum_per_edge  # [E]
            
            # 5. 应用dropout
            attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
            
            # 6. 使用稀疏矩阵乘法聚合邻居特征
            # 构建稀疏注意力矩阵并乘以特征矩阵
            attention_sparse = torch.sparse_coo_tensor(
                edge_index, attention_weights, (N, N), device=input.device, dtype=input.dtype
            )
            h_prime = torch.sparse.mm(attention_sparse, h)  # [N, out_features]
        else:
            # 密集矩阵版本（用于小图或调试）
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
            e = F.leaky_relu(torch.mm(a_input, self.a), negative_slope=self.alpha).view(N, N)
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, p=self.dropout, training=self.training)
            h_prime = torch.mm(attention, h)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class LightweightGAT(nn.Module):
    """
    轻量级图注意力网络 - 显存友好版本
    使用类似GCN的稀疏矩阵乘法，但添加可学习的边权重
    比标准GAT更节省显存，同时保留注意力机制的核心思想
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.2, use_residual=True):
        super(LightweightGAT, self).__init__()
        self.dropout = dropout
        self.use_residual = use_residual
        
        # 第一层：特征变换 + 注意力权重
        self.W1 = Parameter(torch.FloatTensor(nfeat, nhid))
        self.a1 = Parameter(torch.FloatTensor(2 * nhid, 1))
        
        # 第二层：输出层
        self.W2 = Parameter(torch.FloatTensor(nhid, nclass))
        self.a2 = Parameter(torch.FloatTensor(2 * nclass, 1))
        
        # 残差投影
        if use_residual and nfeat != nclass:
            self.res_proj = nn.Linear(nfeat, nclass)
        else:
            self.res_proj = None
        self.ln = nn.LayerNorm(nclass)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv1 = 1. / (self.W1.size(1) ** 0.5)
        self.W1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / (self.a1.size(0) ** 0.5)
        self.a1.data.uniform_(-stdv2, stdv2)
        
        stdv3 = 1. / (self.W2.size(1) ** 0.5)
        self.W2.data.uniform_(-stdv3, stdv3)
        stdv4 = 1. / (self.a2.size(0) ** 0.5)
        self.a2.data.uniform_(-stdv4, stdv4)
    
    def forward(self, x, adj):
        """
        Args:
            x: [N, nfeat] 节点特征
            adj: 稀疏邻接矩阵 [N, N] (支持有向图)
        Returns:
            output: [N, nclass] 输出特征
        """
        identity = x
        N = x.size(0)
        
        if adj.is_sparse:
            # 第一层：特征变换
            h1 = torch.mm(x, self.W1)  # [N, nhid]
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            
            # 计算边的注意力权重（简化版：只计算一次，然后应用到邻接矩阵）
            adj_coo = adj.coalesce()
            edge_index = adj_coo.indices()
            row, col = edge_index[0], edge_index[1]
            
            # 计算注意力分数
            h_row = h1[row]  # [E, nhid]
            h_col = h1[col]  # [E, nhid]
            edge_feat = torch.cat([h_row, h_col], dim=1)  # [E, 2*nhid]
            att_scores = F.leaky_relu(torch.mm(edge_feat, self.a1).squeeze(), negative_slope=0.2)  # [E]
            
            # 使用sigmoid作为简化的注意力权重（避免softmax的复杂计算）
            att_weights = torch.sigmoid(att_scores)  # [E]
            att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
            
            # 构建加权邻接矩阵并聚合
            weighted_adj = torch.sparse_coo_tensor(
                edge_index, att_weights * adj_coo.values(), (N, N),
                device=x.device, dtype=x.dtype
            )
            h1_out = torch.sparse.mm(weighted_adj, h1)  # [N, nhid]
            h1_out = F.relu(h1_out)
            h1_out = F.dropout(h1_out, p=self.dropout, training=self.training)
            
            # 第二层：输出层
            h2 = torch.mm(h1_out, self.W2)  # [N, nclass]
            
            # 再次应用注意力（可选，如果显存允许）
            # 为了节省显存，这里简化处理
            h2_out = torch.sparse.mm(weighted_adj, h2)  # [N, nclass]
            h2_out = F.relu(h2_out)
        else:
            # 密集矩阵版本（小图用）
            h1 = torch.mm(x, self.W1)
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            h1_out = F.relu(torch.mm(adj, h1))
            h1_out = F.dropout(h1_out, p=self.dropout, training=self.training)
            h2 = torch.mm(h1_out, self.W2)
            h2_out = F.relu(torch.mm(adj, h2))
        
        # 残差连接
        if self.use_residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            h2_out = h2_out + identity
        
        h2_out = self.ln(h2_out)
        return h2_out

class GAT(nn.Module):
    """
    图注意力网络，类似GCN的结构但使用注意力机制
    支持有向图，可以学习邻居的重要性权重
    
    注意：标准GAT可能消耗较多显存，如果显存不足，建议使用LightweightGAT
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.2, nheads=1, use_residual=True, lightweight=False):
        super(GAT, self).__init__()
        self.lightweight = lightweight
        
        if lightweight:
            # 使用轻量级版本
            self.gat = LightweightGAT(nfeat, nhid, nclass, dropout, use_residual)
        else:
            # 标准GAT
            self.dropout = dropout
            self.use_residual = use_residual
            
            # 多头注意力（如果nheads=1就是单头）
            self.attentions = nn.ModuleList([
                GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=0.2, concat=True)
                for _ in range(nheads)
            ])
            
            # 第二层注意力（输出层）
            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=0.2, concat=False)
            
            # 残差投影（如果输入输出维度不同）
            if use_residual and nfeat != nclass:
                self.res_proj = nn.Linear(nfeat, nclass)
            else:
                self.res_proj = None
            self.ln = nn.LayerNorm(nclass)
    
    def forward(self, x, adj):
        if self.lightweight:
            return self.gat(x, adj)
        else:
            identity = x
            
            # 第一层：多头注意力
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # [N, nhid * nheads]
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 第二层：输出层
            x = self.out_att(x, adj)  # [N, nclass]
            x = F.relu(x)
            
            # 残差连接
            if self.use_residual:
                if self.res_proj is not None:
                    identity = self.res_proj(identity)
                x = x + identity
            
            x = self.ln(x)
            return x

# ================= 路径内聚合 =================
class IntraPathAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='gru', dropout=0.5, max_path_len=10, use_checkpoint=False):
        super(IntraPathAggregator, self).__init__()
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
            
        self.linear = nn.Linear(2 * (hidden_dim // 2), hidden_dim)
        # [新增] LayerNorm 稳定训练
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, path_features):
        def _rnn_forward(x):
            return self.rnn(x)[0]
        if self.use_checkpoint and self.training:
            output = torch.utils.checkpoint.checkpoint(_rnn_forward, path_features)
        else:
            output, _ = self.rnn(path_features)
        
        path_embedding = torch.mean(output, dim=1) 
        
        path_embedding = self.linear(path_embedding) 
        path_embedding = self.ln(path_embedding)  # [新增] LayerNorm
        path_embedding = F.gelu(path_embedding)
        path_embedding = F.dropout(path_embedding, p=self.dropout, training=self.training)
        
        return path_embedding

# ================= 路径间聚合 =================
class InterPathAggregator(nn.Module):
    def __init__(self, hidden_dim, attn_vec_dim, dropout=0.5):
        super(InterPathAggregator, self).__init__()
        self.attn_vec = nn.Parameter(torch.randn(1, attn_vec_dim))
        self.linear = nn.Linear(hidden_dim, attn_vec_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = dropout

    def forward(self, path_embeddings, num_samples):
        batch_size = path_embeddings.shape[0] // num_samples
        hidden_dim = path_embeddings.shape[1]
        
        embeds = path_embeddings.view(batch_size, num_samples, hidden_dim)
        
        t = torch.tanh(self.linear(embeds)) 
        scores = torch.matmul(t, self.attn_vec.transpose(0, 1))
        scores = self.softmax(scores) 
        
        aggr_embedding = torch.sum(scores * embeds, dim=1)
        return aggr_embedding


# ================= 3. 主模型 (MAGNN + KT) =================
class MAGNN_KT(nn.Module):
    def __init__(self, num_nodes_list, input_dim, hidden_dim, 
                 metapath_list, num_samples, device, dropout=0.5, question_skill_info=None,
                 magnn_chunk_size=4096, use_checkpoint=False,
                 ablate_difficulty=False, ablate_interaction=False,
                 upper_encoder='gru', upper_tf_layers=2, upper_tf_heads=2, upper_tf_ffn_mul=4, max_seq_len=200):
        super(MAGNN_KT, self).__init__()
        self.device = device
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        mp_map = {
            "0-1-0": [0, 1, 0],
            "0-2-0": [0, 2, 0],
            "0-1-3-1-0": [0, 1, 3, 1, 0]
        }
        self.mp_types = [mp_map[mp] for mp in metapath_list]

        # --- A. 特征嵌入层 ---
        self.num_questions = num_nodes_list[0]
        self.embeddings = nn.ModuleList()
        for num_nodes in num_nodes_list:
            self.embeddings.append(nn.Embedding(num_nodes, input_dim))
        
        self.difficulty_embed = nn.Embedding(num_nodes_list[0], 1)
        nn.init.constant_(self.difficulty_embed.weight, 0.0)
        # 初始化为 0 (假设一开始大家难度都一样，让模型去学差异)
        self.feat_projs = nn.ModuleList()
        for _ in num_nodes_list:
            self.feat_projs.append(nn.Linear(input_dim, hidden_dim))
            
        # --- B. MAGNN 编码层 (恢复) ---
        self.intra_aggregators = nn.ModuleList()
        self.inter_aggregators = nn.ModuleList()
        attn_dim = min(128, hidden_dim)
        
        for _ in metapath_list:
            # 这里的输入维度是 input_dim 投影后的 hidden_dim
            self.intra_aggregators.append(IntraPathAggregator(hidden_dim, hidden_dim, dropout=self.dropout, max_path_len=10, use_checkpoint=use_checkpoint))
            self.inter_aggregators.append(InterPathAggregator(hidden_dim, attn_dim))
            
        self.metapath_attn_vec = nn.Parameter(torch.randn(1, attn_dim))
        self.metapath_linear = nn.Linear(hidden_dim, attn_dim)
        self.self_gate = nn.Linear(hidden_dim * 2, 1)
        self.use_skill = False
        if question_skill_info is not None:
            skill_indices, skill_offsets, num_skills = question_skill_info
            self.use_skill = True
            self.skill_bag = nn.EmbeddingBag(num_skills, hidden_dim, mode="mean", include_last_offset=True)
            self.register_buffer("skill_indices", skill_indices.long())
            self.register_buffer("skill_offsets", skill_offsets.long())
            self.skill_gate = nn.Linear(hidden_dim * 2, 1)
        
        # --- C. KT 预测层 ---
        self.upper_encoder = upper_encoder
        if self.upper_encoder == 'gru':
            self.kt_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
            self.kt_tf = None
            self.pos_embed = None
        else:
            if hidden_dim % upper_tf_heads != 0:
                raise ValueError("hidden_dim must be divisible by upper_tf_heads")
            ffn_dim = hidden_dim * upper_tf_ffn_mul
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=upper_tf_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout * 0.5,
                batch_first=True,
                activation="gelu",
                norm_first=True
            )
            self.kt_tf = nn.TransformerEncoder(encoder_layer, num_layers=upper_tf_layers)
            self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
            self.kt_rnn = None
        
        # [新增] 预测层改为小MLP，增强表达能力
        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(self.dropout * 0.5),  # 较小的dropout
            nn.Linear(hidden_dim, 1)
        )
        # 残差投影层（如果输入输出维度不同时使用）
        self.rnn_residual = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.ans_embed = nn.Embedding(2, hidden_dim)
        self.interaction_embed = nn.Embedding(2 * self.num_questions, hidden_dim)
        self.diff_scale = nn.Parameter(torch.tensor(0.0))
        self.magnn_chunk_size = magnn_chunk_size
        
        # [可靠改进] LayerNorm 稳定训练
        self.magnn_ln = nn.LayerNorm(hidden_dim)
        self.rnn_ln = nn.LayerNorm(hidden_dim)
        self.rnn_input_ln = nn.LayerNorm(hidden_dim)
        
        # 消融开关
        self.ablate_difficulty = ablate_difficulty
        self.ablate_interaction = ablate_interaction

    def get_magnn_embedding(self, q_ids, processed_data):
        features_list = processed_data[2] 
        idx_lists = processed_data[1] 
        sem_feats = self.embeddings[0](q_ids)
        target_q_feats = torch.tanh(self.feat_projs[0](sem_feats))
        
        metapath_embeds = []
        
        for i, (intra_agg, inter_agg) in enumerate(zip(self.intra_aggregators, self.inter_aggregators)):
            current_path_len = len(self.mp_types[i])
            
            batch_paths = []
            
            for q in q_ids:
                q_val = q.item()
                pool = idx_lists[i][q_val] 
                
                # --- 核心：保留之前的强力形状修复逻辑 ---
                
                if len(pool) == 0:
                    selected_paths = np.zeros((self.num_samples, current_path_len), dtype=np.int64)
                elif len(pool) == self.num_samples and pool.shape[1] == current_path_len:
                    selected_paths = pool
                else:
                    if len(pool) >= self.num_samples:
                        indices = np.random.choice(len(pool), self.num_samples, replace=False)
                    else:
                        indices = np.random.choice(len(pool), self.num_samples, replace=True)
                    
                    selected_paths = pool[indices]

                    if selected_paths.shape[1] != current_path_len:
                        if selected_paths.ndim == 1:
                            selected_paths = selected_paths.reshape(self.num_samples, -1)

                # D. 强制类型转换 (防止 Object 类型)
                batch_paths.append(selected_paths.astype(np.int64))
            
            # --- 堆叠与转换 ---
            try:
                batch_paths_np = np.stack(batch_paths) 
                batch_paths_tensor = torch.from_numpy(batch_paths_np).to(self.device) # [Batch, Samples, Path_Len]
            except Exception as e:
                print(f"\n[Stack Error] MetaPath {i} stack failed.")
                raise e
            
            # --- Feature Lookup (恢复为完整路径查找) ---
            path_feats = self._lookup_path_features(batch_paths_tensor, i)
            
            # Flatten 
            b, s, l, h = path_feats.shape
            path_feats = path_feats.view(b*s, l, h)
            
            # Aggregation (走 RNN) with chunking to reduce memory footprint
            total_items = path_feats.shape[0]
            chunks = []
            for start in range(0, total_items, self.magnn_chunk_size):
                end = min(start + self.magnn_chunk_size, total_items)
                chunk_out = intra_agg(path_feats[start:end])
                chunks.append(chunk_out)
            h_intra = torch.cat(chunks, dim=0)
            h_inter = inter_agg(h_intra, self.num_samples)
            
            metapath_embeds.append(h_inter)
            
        stack_embeds = torch.stack(metapath_embeds, dim=1)
        mp_proj = self.metapath_linear(stack_embeds)
        
        t = torch.tanh(mp_proj)
        
        scores = torch.matmul(t, self.metapath_attn_vec.transpose(0, 1))
        scores = F.softmax(scores, dim=1)
        
        q_final_embed = torch.sum(scores * stack_embeds, dim=1) 
        gate_in = torch.cat([q_final_embed, target_q_feats], dim=-1)
        gate = torch.sigmoid(self.self_gate(gate_in))
        q_final_embed = gate * q_final_embed + (1 - gate) * target_q_feats
        if self.use_skill:
            skill_embed = self.skill_bag(self.skill_indices, self.skill_offsets)
            q_skill = skill_embed[q_ids]
            skill_gate = torch.sigmoid(self.skill_gate(torch.cat([q_final_embed, q_skill], dim=-1)))
            q_final_embed = skill_gate * q_final_embed + (1 - skill_gate) * q_skill
        
        return q_final_embed

    def _lookup_path_features(self, batch_paths, mp_index):
        # 【恢复】循环读取路径上每一个节点的特征
        curr_types = self.mp_types[mp_index]
        path_len = batch_paths.shape[2]
        feats = []
        
        for pos in range(path_len):
            node_type = curr_types[pos]
            node_ids = batch_paths[:, :, pos] 
            base_feat = self.embeddings[node_type](node_ids)
            node_emb = torch.tanh(self.feat_projs[node_type](base_feat))
            
            feats.append(node_emb)
            
        return torch.stack(feats, dim=2)

    def forward(self, q_seqs, a_seqs, processed_data, mask=None):
        batch_size, seq_len = q_seqs.shape
        flat_q = q_seqs.view(-1) 
        
        graph_embeds = self.get_magnn_embedding(flat_q, processed_data)
        q_embeds = graph_embeds.view(batch_size, seq_len, -1)
        
        # [新增] MAGNN输出加LayerNorm
        q_embeds = self.magnn_ln(q_embeds)
        
        a_embeds = self.ans_embed(a_seqs) 
        inter_idx = q_seqs + (1 - a_seqs) * self.num_questions
        inter_embeds = self.interaction_embed(inter_idx)
        
        if mask is not None:
            m = mask.unsqueeze(-1)
            q_embeds = q_embeds * m
            a_embeds = a_embeds * m
            inter_embeds = inter_embeds * m
        
        # 消融：交互嵌入
        if self.ablate_interaction:
            rnn_input = q_embeds + a_embeds  # 不使用交互嵌入
        else:
            rnn_input = q_embeds + a_embeds + inter_embeds
        
        rnn_input = self.rnn_input_ln(rnn_input)
        
        if self.upper_encoder == 'gru':
            h_out, _ = self.kt_rnn(rnn_input)
        else:
            seq_len = rnn_input.size(1)
            positions = torch.arange(seq_len, device=rnn_input.device).unsqueeze(0).expand(batch_size, seq_len)
            pos_emb = self.pos_embed(positions)
            if mask is not None:
                pos_emb = pos_emb * mask.unsqueeze(-1)
            rnn_input = rnn_input + pos_emb
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = mask == 0
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=rnn_input.device, dtype=torch.bool), diagonal=1)
            h_out = self.kt_tf(rnn_input, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        # [可靠改进] RNN输出加LayerNorm
        h_out = self.rnn_ln(h_out)
        # 残差连接：增强梯度流动
        h_out = h_out + self.rnn_residual(rnn_input)
        
        h_t_forward = h_out[:, :-1, :] 
        q_next = q_embeds[:, 1:, :]
        
        cat_feature = torch.cat([h_t_forward, q_next], dim=-1) 
        cat_feature = self.dropout_layer(cat_feature)
        ability_logits = self.pred_layer(cat_feature).squeeze(-1)
        
        target_q = q_seqs[:, 1:] # [Batch, Seq-1]
        
        # 2. 查表得到这道题的难度
        diff_raw = self.difficulty_embed(target_q).squeeze(-1) # [Batch, Seq-1]
        diff = diff_raw * self.diff_scale
        
        # 3. 核心公式: 预测值 = 能力 - 难度
        # 能力强、题目简单(diff小) -> logits大 -> 预测对
        # 能力弱、题目难(diff大) -> logits小 -> 预测错
        final_logits = ability_logits - diff
        
        labels = a_seqs[:, 1:].float()
        
        # 返回修正后的 logits
        return final_logits, labels

class MAGNN_DGCN_Fusion(MAGNN_KT):
    def __init__(self, num_nodes_list, input_dim, hidden_dim, metapath_list, num_samples, device, num_questions, dropout=0.5, question_skill_info=None, magnn_chunk_size=4096, use_checkpoint=False, fusion_mode='gate', ablate_difficulty=False, ablate_skill=False, ablate_interaction=False, gat_dropout=0.2, gat_nheads=1, upper_encoder='gru', upper_tf_layers=2, upper_tf_heads=2, upper_tf_ffn_mul=4, max_seq_len=200):
        super(MAGNN_DGCN_Fusion, self).__init__(num_nodes_list, input_dim, hidden_dim, metapath_list, num_samples, device, dropout=dropout, question_skill_info=(None if ablate_skill else question_skill_info), magnn_chunk_size=magnn_chunk_size, use_checkpoint=use_checkpoint, ablate_difficulty=ablate_difficulty, ablate_interaction=ablate_interaction, upper_encoder=upper_encoder, upper_tf_layers=upper_tf_layers, upper_tf_heads=upper_tf_heads, upper_tf_ffn_mul=upper_tf_ffn_mul, max_seq_len=max_seq_len)
        self.num_questions = num_questions
        self.fusion_mode = fusion_mode
        # 消融开关
        self.ablate_difficulty = ablate_difficulty
        self.ablate_skill = ablate_skill
        self.ablate_interaction = ablate_interaction
        self.register_buffer("dgc_node_ids", torch.arange(2 * num_questions, dtype=torch.long))
        self.dgc_emb = nn.Embedding(2 * num_questions, hidden_dim)
        self.dgc_sem_alpha = nn.Parameter(torch.tensor(0.5))
        # 【改进】使用轻量级GAT替代GCN，更好地处理有向图和邻居重要性，同时节省显存
        # 轻量级GAT使用类似GCN的稀疏矩阵乘法，但添加可学习的边权重
        # 显存占用远低于标准GAT，适合16GB显存
        self.dgc_out = GAT(nfeat=hidden_dim, nhid=hidden_dim, nclass=hidden_dim, dropout=gat_dropout, nheads=gat_nheads, lightweight=True)
        self.dgc_in = GAT(nfeat=hidden_dim, nhid=hidden_dim, nclass=hidden_dim, dropout=gat_dropout, nheads=gat_nheads, lightweight=True)
        # DGC RNN（保持单层）
        self.dgc_rnn = nn.GRU(2 * hidden_dim, hidden_dim, batch_first=True)
        self.dgc_gate = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.dgc_dropout = nn.Dropout(0.3)
        # [可靠改进] LayerNorm
        self.dgc_ln = nn.LayerNorm(2 * hidden_dim)
        self.dgc_rnn_ln = nn.LayerNorm(hidden_dim)
        # DGC 预测层
        self.dgc_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 1)
        )
        
        # =============== 简洁的融合机制（回退版本） ===============
        # 核心思路：MAGNN效果更好(78.60)，让融合更依赖MAGNN
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的融合偏置
        self.fusion_gate = nn.Linear(2, 1)
        # 基于特征的门控
        self.gating_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        if self.fusion_mode == 'concat':
            self.concat_mlp = nn.Sequential(
                nn.Linear(2, 2),
                nn.ReLU(),
                nn.Linear(2, 1)
            )

    def forward(self, q_seqs, a_seqs, processed_data, adj_out, adj_in, mask=None, return_branch=False, compute_branch='both'):
        labels = a_seqs[:, 1:].float()
        mag_logits = None
        dgc_logits = None
        if compute_branch in ('both', 'mag'):
            mag_logits, _labels = super().forward(q_seqs, a_seqs, processed_data, mask=mask)
            labels = _labels
        if compute_branch in ('both', 'dgc'):
            if torch.is_autocast_enabled():
                with torch.cuda.amp.autocast(enabled=False):
                    sem_q = torch.tanh(self.feat_projs[0](self.embeddings[0].weight)).float()
                    sem_pair = torch.cat([sem_q, sem_q], dim=0)
                    beta = torch.sigmoid(self.dgc_sem_alpha).float()
                    ques_init = self.dgc_emb(self.dgc_node_ids).float() + beta * sem_pair
                    ques_out = self.dgc_out(ques_init, adj_out.float())
                    ques_in = self.dgc_in(ques_init, adj_in.float())
                    ques_d = torch.cat([ques_in, ques_out], -1)
            else:
                sem_q = torch.tanh(self.feat_projs[0](self.embeddings[0].weight))
                sem_pair = torch.cat([sem_q, sem_q], dim=0)
                beta = torch.sigmoid(self.dgc_sem_alpha)
                ques_init = self.dgc_emb(self.dgc_node_ids) + beta * sem_pair
                ques_out = self.dgc_out(ques_init, adj_out)
                ques_in = self.dgc_in(ques_init, adj_in)
                ques_d = torch.cat([ques_in, ques_out], -1)
            incorrect = (1 - a_seqs).long()
            inter_idx = q_seqs + incorrect * self.num_questions
            x_d = ques_d[inter_idx]
            a_emb = self.ans_embed(a_seqs)
            g = torch.sigmoid(self.dgc_gate(a_emb))
            x_d = x_d * g
            if mask is not None:
                x_d = x_d * mask.unsqueeze(-1)
            x_d = self.dgc_ln(x_d)
            x_d = self.dgc_dropout(x_d)
            h_out, _ = self.dgc_rnn(x_d)
            # [可靠改进] DGC RNN输出加LayerNorm
            h_out = self.dgc_rnn_ln(h_out)
            h_t = h_out[:, :-1, :]
            ques_base = (ques_d[:self.num_questions] + ques_d[self.num_questions:]) / 2
            q_next = ques_base[q_seqs[:, 1:]]
            cat_feature = torch.cat([h_t, q_next], dim=-1)
            dgc_logits = self.dgc_mlp(cat_feature).squeeze(-1)
            target_q = q_seqs[:, 1:]
            diff_raw = self.difficulty_embed(target_q).squeeze(-1)
            diff = diff_raw * self.diff_scale
            dgc_logits = dgc_logits - diff
        if compute_branch == 'mag':
            if return_branch:
                return mag_logits, labels, mag_logits, None
            return mag_logits, labels
        if compute_branch == 'dgc':
            if return_branch:
                return dgc_logits, labels, None, dgc_logits
            return dgc_logits, labels
        if mag_logits is None or dgc_logits is None:
            raise ValueError("compute_branch=both 但某一分支未计算")
        
        gate_in = torch.stack([mag_logits, dgc_logits], dim=-1)
        
        if self.fusion_mode == 'concat':
            fused_logits = self.concat_mlp(gate_in).squeeze(-1)
        else:
            # 简洁的融合：logits门控 + 特征门控（回退到原始机制）
            alpha_h = torch.sigmoid(self.gating_mlp(cat_feature).squeeze(-1))
            alpha = torch.sigmoid(self.fusion_gate(gate_in).squeeze(-1) + self.fusion_alpha + alpha_h - 0.5)
            fused_logits = alpha * mag_logits + (1 - alpha) * dgc_logits
        
        if return_branch:
            return fused_logits, labels, mag_logits, dgc_logits
        return fused_logits, labels
