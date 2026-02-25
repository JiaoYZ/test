import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
BASE_DIR = r"D:\Study\Study_Paper\code\ex\ttt\aaa\kt\ednet"
MAPPED_DIR = os.path.join(BASE_DIR, "mapped_data")
OUTPUT_DIR = os.path.join(MAPPED_DIR, "instances")

# 输入文件
STU_QUES_PATH = os.path.join(MAPPED_DIR, "stu_ques_mapped.csv")
STU_CLUSTER_PATH = os.path.join(MAPPED_DIR, "stu_cluster_8_mapped.csv")

# 输出路径
SAVE_PATH = os.path.join(OUTPUT_DIR, "qucuq_paths_1000.npy")

# 采样设置
NUM_WALKS = 100
BATCH_SIZE = 256
D = 1.0
SAMPLING_MODE = os.getenv("INSTANCE_SAMPLING", "biased")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def to_tensor(x):
    return torch.tensor(x).to(DEVICE)

class BiasedWalker:
    """
    原版 BiasedWalker (保留 Edge Weight Consistency)
    """
    def __init__(self, v_num, edges, use_edge_weight=False, D=1.0, sampling_mode="biased"):
        self.v_num = v_num
        self.use_edge_weight = use_edge_weight
        self.D = D
        self.sampling_mode = sampling_mode

        ngbrs = [[] for _ in range(v_num)]
        weights = [[] for _ in range(v_num)]

        for u, v, w in edges:
            u, v = int(u), int(v)
            ngbrs[u].append(v)
            weights[u].append(w)

        for i in range(v_num):
            if len(ngbrs[i]) == 0:
                ngbrs[i].append(i)
                weights[i].append(0)

        self.ngbrs = pad_sequence([to_tensor(s) for s in ngbrs], batch_first=True, padding_value=v_num)
        self.weights = pad_sequence([torch.tensor(s, dtype=torch.float32).to(DEVICE) for s in weights], 
                                   batch_first=True, padding_value=0)
        
        self.ngbr_nums = to_tensor([len(n) for n in ngbrs]).unsqueeze(1)
        self.max_ngbr_num = self.ngbrs.shape[1]

    def get_prob(self, weights, nums, prev_weights=None):
        batch_size = weights.shape[0]
        if self.sampling_mode == "uniform":
            prob = torch.ones(weights.size()).to(DEVICE)
        elif self.use_edge_weight and prev_weights is not None:
            diff = torch.abs(weights - prev_weights.unsqueeze(1))
            prob = 1.0 - torch.true_divide(diff, self.D)
            prob = torch.clamp(prob, min=1e-6)
        else:
            prob = torch.ones(weights.size()).to(DEVICE)

        mask_indices = torch.arange(self.max_ngbr_num).expand(batch_size, -1).to(DEVICE)
        mask = mask_indices >= nums
        prob = prob.masked_fill(mask, 0)

        row_sum = torch.sum(prob, dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        prob = torch.true_divide(prob, row_sum)

        return prob

    def sample_next(self, current_nodes, prev_weights=None):
        expand_weights = self.weights[current_nodes]
        expand_ngbrs = self.ngbrs[current_nodes]
        expand_nums = self.ngbr_nums[current_nodes]

        prob = self.get_prob(expand_weights, expand_nums, prev_weights)
        next_indices = torch.multinomial(prob, num_samples=1)
        
        next_nodes = torch.gather(expand_ngbrs, 1, next_indices).squeeze(1)
        current_step_weights = torch.gather(expand_weights, 1, next_indices).squeeze(1)
        return next_nodes, current_step_weights


def generate_qucuq_instances():
    print(f"Instance sampling mode: {SAMPLING_MODE}")
    print("Loading Data...")
    stu_ques_df = pd.read_csv(STU_QUES_PATH)
    stu_cluster_df = pd.read_csv(STU_CLUSTER_PATH)
    
    # 获取节点总数 (直接用Max ID + 1)
    # stu_cluster_mapped.csv 的列名可能是 'stu' 或 'user_id'，需要判断
    if 'stu' in stu_cluster_df.columns:
        num_stu = max(stu_ques_df['user_id'].max(), stu_cluster_df['stu'].max()) + 1
    elif 'user_id' in stu_cluster_df.columns:
        num_stu = max(stu_ques_df['user_id'].max(), stu_cluster_df['user_id'].max()) + 1
    else:
        raise ValueError(f"Unknown column names in stu_cluster_mapped.csv: {stu_cluster_df.columns.tolist()}")
    
    num_ques = stu_ques_df['problem_id'].max() + 1
    num_cluster = stu_cluster_df['cluster'].max() + 1
    
    print(f"Stats: Stu={num_stu}, Ques={num_ques}, Cluster={num_cluster}")
    
    print("Building edge lists...")
    qu_edges = []
    uq_edges = []
    uc_edges = []
    cu_edges = []

    # Q <-> U
    for _, row in stu_ques_df.iterrows():
        s, q, correct = int(row['user_id']), int(row['problem_id']), int(row['correct'])
        qu_edges.append((q, s, correct))
        uq_edges.append((s, q, correct))
    
    # U <-> C
    # 根据实际列名确定使用哪个列名
    for _, row in stu_cluster_df.iterrows():
        if 'stu' in stu_cluster_df.columns:
            s, c = int(row['stu']), int(row['cluster'])
        elif 'user_id' in stu_cluster_df.columns:
            s, c = int(row['user_id']), int(row['cluster'])
        else:
            raise ValueError(f"Unknown column names in stu_cluster_mapped.csv: {stu_cluster_df.columns.tolist()}")
        uc_edges.append((s, c, 1))
        cu_edges.append((c, s, 1))
    
    print("Initializing Walkers...")
    QU_Walker = BiasedWalker(num_ques, qu_edges, use_edge_weight=False, D=D, sampling_mode=SAMPLING_MODE)
    UC_Walker = BiasedWalker(num_stu, uc_edges, use_edge_weight=False, D=D, sampling_mode=SAMPLING_MODE)
    CU_Walker = BiasedWalker(num_cluster, cu_edges, use_edge_weight=False, D=D, sampling_mode=SAMPLING_MODE)
    UQ_Walker = BiasedWalker(num_stu, uq_edges, use_edge_weight=True, D=D, sampling_mode=SAMPLING_MODE)
    
    print(f"Generating instances: {NUM_WALKS} walks per question...")
    all_paths = []
    
    total_ques_range = num_ques
    for batch_start in range(0, total_ques_range, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_ques_range)
        print(f"Processing questions {batch_start} to {batch_end}...", end='\r')
        
        batch_ques = torch.arange(batch_start, batch_end).to(DEVICE)
        start_nodes = batch_ques.repeat_interleave(NUM_WALKS)
        
        # Q -> U
        u1_nodes, w_qu = QU_Walker.sample_next(start_nodes, prev_weights=None)
        # U -> C
        c_nodes, _ = UC_Walker.sample_next(u1_nodes, prev_weights=None)
        # C -> U
        u2_nodes, _ = CU_Walker.sample_next(c_nodes, prev_weights=None)
        # U -> Q (Consistency)
        end_q_nodes, _ = UQ_Walker.sample_next(u2_nodes, prev_weights=w_qu)
        
        paths = torch.stack([start_nodes, u1_nodes, c_nodes, u2_nodes, end_q_nodes], dim=1)
        all_paths.append(paths.cpu())
        torch.cuda.empty_cache()
    
    paths = torch.cat(all_paths, dim=0)
    print(f"\nSaving {paths.shape[0]} instances to {SAVE_PATH}...")
    np.save(SAVE_PATH, paths.numpy())
    print("Done!")

if __name__ == "__main__":
    generate_qucuq_instances()
