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

# 输入文件 (已映射)
STU_QUES_PATH = os.path.join(MAPPED_DIR, "stu_ques_mapped.csv")
QUES_DISC_PATH = os.path.join(MAPPED_DIR, "ques_discvalue_mapped.csv")
STU_ABI_PATH = os.path.join(MAPPED_DIR, "stu_abi_mapped.csv")

# 输出路径
SAVE_PATH = os.path.join(OUTPUT_DIR, "quq_paths_1000.npy")

# 采样设置
NUM_WALKS = 100
BATCH_SIZE = 256
SAMPLING_MODE = os.getenv("INSTANCE_SAMPLING", "biased")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def to_tensor(x):
    return torch.tensor(x).to(DEVICE)

class BiasedWalker:
    """
    原版 BiasedWalker (保留 Node Attribute 偏置逻辑)
    """
    def __init__(self, v_num, edges, node_attr=None, D=1.0, sampling_mode="biased"):
        self.v_num = v_num
        self.node_attr = node_attr
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

    def get_prob(self, weights, ngbrs, nums, prev_weights=None):
        if self.sampling_mode == "uniform":
            prob = torch.ones(weights.size()).to(DEVICE)
        elif prev_weights is not None:
            diff = torch.abs(weights - prev_weights.unsqueeze(1))
            prob = 1.0 - torch.true_divide(diff, self.D)
            prob = torch.clamp(prob, min=1e-6)
        else:
            prob = torch.ones(weights.size()).to(DEVICE)
        if self.sampling_mode != "uniform" and self.node_attr is not None:
            safe_ngbrs = ngbrs.clone()
            mask_pad = safe_ngbrs >= self.node_attr.size(0)
            safe_ngbrs[mask_pad] = 0 
            attr_prob = F.embedding(safe_ngbrs, self.node_attr).squeeze(-1)
            prob = prob * attr_prob
        mask_indices = torch.arange(self.max_ngbr_num).expand(weights.shape[0], -1).to(DEVICE)
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

        prob = self.get_prob(expand_weights, expand_ngbrs, expand_nums, prev_weights)
        next_indices = torch.multinomial(prob, num_samples=1)
        
        next_nodes = torch.gather(expand_ngbrs, 1, next_indices).squeeze(1)
        current_step_weights = torch.gather(expand_weights, 1, next_indices).squeeze(1)
        return next_nodes, current_step_weights

def generate_quq_instances():
    print(f"Instance sampling mode: {SAMPLING_MODE}")
    print("Loading Data...")
    stu_ques_df = pd.read_csv(STU_QUES_PATH) # columns: user_id, problem_id, correct
    ques_disc_df = pd.read_csv(QUES_DISC_PATH) # columns: ques_id, discrimination
    stu_abi_df = pd.read_csv(STU_ABI_PATH) # columns: 可能是 stu_id 或 user_id, ability
    
    # 确定节点数量 (使用 CSV 中的最大ID，因为数据已是 0,1,2...)
    # 根据实际列名确定使用哪个列名
    if 'stu_id' in stu_abi_df.columns:
        num_stu = max(stu_ques_df['user_id'].max(), stu_abi_df['stu_id'].max()) + 1
    elif 'user_id' in stu_abi_df.columns:
        num_stu = max(stu_ques_df['user_id'].max(), stu_abi_df['user_id'].max()) + 1
    else:
        raise ValueError(f"Unknown column names in stu_abi_mapped.csv: {stu_abi_df.columns.tolist()}")
    
    # 根据实际列名确定使用哪个列名
    if 'ques_id' in ques_disc_df.columns:
        num_ques = max(stu_ques_df['problem_id'].max(), ques_disc_df['ques_id'].max()) + 1
    elif 'problem_id' in ques_disc_df.columns:
        num_ques = max(stu_ques_df['problem_id'].max(), ques_disc_df['problem_id'].max()) + 1
    else:
        raise ValueError(f"Unknown column names in ques_discvalue_mapped.csv: {ques_disc_df.columns.tolist()}")
    
    print(f"Nodes: Stu={num_stu}, Ques={num_ques}")
    
    # --- 准备节点属性 (Attributes) ---
    print("Preparing attributes...")
    # 学生能力
    # 创建全零数组，然后填入对应 ID 的值
    ability_array = np.zeros(num_stu)
    # 根据实际列名确定使用哪个列名
    for _, row in stu_abi_df.iterrows():
        if 'stu_id' in stu_abi_df.columns:
            ability_array[int(row['stu_id'])] = row['ability']
        elif 'user_id' in stu_abi_df.columns:
            ability_array[int(row['user_id'])] = row['ability']
        else:
            raise ValueError(f"Unknown column names in stu_abi_mapped.csv: {stu_abi_df.columns.tolist()}")
        
    abi_avg = np.average(ability_array[ability_array != 0]) # 计算非零平均
    ability_tensor = torch.tensor(ability_array, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    stu_attr_mat = torch.sigmoid(-torch.abs(ability_tensor - abi_avg))
    
    # 题目区分度
    disc_array = np.zeros(num_ques)
    # 根据实际列名确定使用哪个列名
    for _, row in ques_disc_df.iterrows():
        if 'ques_id' in ques_disc_df.columns:
            disc_array[int(row['ques_id'])] = row['discrimination']
        elif 'problem_id' in ques_disc_df.columns:
            disc_array[int(row['problem_id'])] = row['discrimination']
        else:
            raise ValueError(f"Unknown column names in ques_discvalue_mapped.csv: {ques_disc_df.columns.tolist()}")
        
    disc_tensor = torch.tensor(disc_array, dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    ques_attr_mat = torch.sigmoid(disc_tensor)

    # --- 准备边 (Edges) ---
    qu_edge_list = []
    uq_edge_list = []

    # stu_ques_mapped 的列名是 user_id, problem_id, correct
    for _, row in stu_ques_df.iterrows():
        s, q, correct = int(row['user_id']), int(row['problem_id']), int(row['correct'])
        qu_edge_list.append((q, s, correct))
        uq_edge_list.append((s, q, correct))

    print("Initializing Walkers...")
    # Add Padding for attributes
    stu_attr_padded = torch.cat([stu_attr_mat, torch.zeros(1, 1).to(DEVICE)], dim=0)
    ques_attr_padded = torch.cat([ques_attr_mat, torch.zeros(1, 1).to(DEVICE)], dim=0)

    QU_Walker = BiasedWalker(num_ques, qu_edge_list, node_attr=stu_attr_padded, sampling_mode=SAMPLING_MODE) 
    UQ_Walker = BiasedWalker(num_stu, uq_edge_list, node_attr=ques_attr_padded, sampling_mode=SAMPLING_MODE)

    print(f"Generating instances: {NUM_WALKS} walks per question...")
    all_paths = []
    
    total_ques_range = num_ques
    for batch_start in range(0, total_ques_range, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_ques_range)
        batch_ques = torch.arange(batch_start, batch_end).to(DEVICE)
        print(f"Processing questions {batch_start} to {batch_end}...", end='\r')
        
        start_nodes = batch_ques.repeat_interleave(NUM_WALKS)

        # Q -> U (Ability Bias)
        next_u, w_qu = QU_Walker.sample_next(start_nodes, prev_weights=None)

        # U -> Q (Discrimination Bias + Consistency)
        end_q, _ = UQ_Walker.sample_next(next_u, prev_weights=w_qu)

        paths = torch.stack([start_nodes, next_u, end_q], dim=1)
        all_paths.append(paths.cpu())
        torch.cuda.empty_cache()
    
    paths = torch.cat(all_paths, dim=0)
    print(f"\nSaving {paths.shape[0]} instances to {SAVE_PATH}...")
    np.save(SAVE_PATH, paths.numpy())
    print("Done.")

if __name__ == "__main__":
    generate_quq_instances()
