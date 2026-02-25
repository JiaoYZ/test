import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
import ast

# ================= 配置区域 =================
BASE_DIR = r"D:\Study\Study_Paper\code\ex\ttt\aaa\kt\ednet"
MAPPED_DIR = os.path.join(BASE_DIR, "mapped_data")
OUTPUT_DIR = os.path.join(MAPPED_DIR, "instances")

# 输入文件 (已映射)
DIFF_DICT_PATH = os.path.join(MAPPED_DIR, "quesID2diffValue_dict_mapped.txt")
QUES_SKILL_CSV_PATH = os.path.join(MAPPED_DIR, "ques_skill_mapped.csv")

# 输出路径
SAVE_PATH = os.path.join(OUTPUT_DIR, "qkq_paths_1000.npy")

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
    原版 BiasedWalker (保留你的逻辑)
    """
    def __init__(self, v_num, edges, sampling_mode="biased"):
        self.v_num = v_num
        self.sampling_mode = sampling_mode
        
        # 1. 构建邻接表和权重表
        ngbrs = [[] for _ in range(v_num)]
        weights = [[] for _ in range(v_num)]
        
        for u, v, w in edges:
            u, v = int(u), int(v)
            ngbrs[u].append(v)
            weights[u].append(w)

        # 处理孤立节点
        for i in range(v_num):
            if len(ngbrs[i]) == 0:
                ngbrs[i].append(i)
                weights[i].append(0.5) 

        self.ngbrs = pad_sequence([to_tensor(s) for s in ngbrs], batch_first=True, padding_value=v_num)
        self.weights = pad_sequence([torch.tensor(s, dtype=torch.float32).to(DEVICE) for s in weights], 
                                   batch_first=True, padding_value=0)
        
        self.ngbr_nums = to_tensor([len(n) for n in ngbrs]).unsqueeze(1)
        self.max_ngbr_num = self.ngbrs.shape[1]
        self.D = 1.0

    def get_prob(self, weights, ngbrs, nums, prev_weights=None):
        if self.sampling_mode == "uniform":
            prob = torch.ones(weights.size()).to(DEVICE)
        elif prev_weights is not None:
            diff = torch.abs(weights - prev_weights.unsqueeze(1))
            prob = 1.0 - diff / self.D
            prob = torch.clamp(prob, min=0.0)
        else:
            prob = torch.ones(weights.size()).to(DEVICE)

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

def generate_qkq_instances():
    print(f"Instance sampling mode: {SAMPLING_MODE}")
    print(f"Loading Difficulty Dict from: {DIFF_DICT_PATH}")
    with open(DIFF_DICT_PATH, 'r') as f:
        ques2diff_dict = ast.literal_eval(f.read())
    
    print(f"Loading Graph from: {QUES_SKILL_CSV_PATH}")
    df = pd.read_csv(QUES_SKILL_CSV_PATH)
    # 根据 remap 后的实际列名确定使用哪个列名
    if 'ques' in df.columns and 'skill' in df.columns:
        # 如果存在 'ques' 和 'skill' 列名（2017 格式）
        num_ques = df['ques'].max() + 1
        num_skill = df['skill'].max() + 1
    elif 'problem_id' in df.columns and 'skill_id' in df.columns:
        # 如果存在 'problem_id' 和 'skill_id' 列名（2009 格式）
        num_ques = df['problem_id'].max() + 1
        num_skill = df['skill_id'].max() + 1
    else:
        raise ValueError(f"Unknown column names in ques_skill_mapped.csv: {df.columns.tolist()}")
    
    print(f"Graph Stats: Questions={num_ques}, Skills={num_skill}")

    # 构建边列表
    qk_edges = [] 
    kq_edges = [] 
    
    for _, row in df.iterrows():
        # 根据实际列名获取ID
        if 'ques' in df.columns and 'skill' in df.columns:
            q_id = int(row['ques'])
            k_id = int(row['skill'])
        elif 'problem_id' in df.columns and 'skill_id' in df.columns:
            q_id = int(row['problem_id'])
            k_id = int(row['skill_id'])
        else:
            raise ValueError(f"Unknown column names in ques_skill_mapped.csv: {df.columns.tolist()}")
        
        diff = ques2diff_dict.get(q_id, 0.5) 
        
        qk_edges.append((q_id, k_id, diff))
        kq_edges.append((k_id, q_id, diff))

    print("Initializing Walkers...")
    QK_Walker = BiasedWalker(num_ques, qk_edges, sampling_mode=SAMPLING_MODE)
    KQ_Walker = BiasedWalker(num_skill, kq_edges, sampling_mode=SAMPLING_MODE)

    print(f"Generating instances: {NUM_WALKS} walks per question...")
    all_paths = []
    
    # 覆盖所有题目 ID (0 到 N-1)
    # 如果题目在 CSV 中没出现（孤立点），这里也要处理，否则 MAGNN 索引会错位
    total_ques_range = num_ques # 使用最大ID作为范围
    
    for batch_start in range(0, total_ques_range, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_ques_range)
        batch_ques = torch.arange(batch_start, batch_end).to(DEVICE)
        print(f"Processing questions {batch_start} to {batch_end}...", end='\r')
        
        start_nodes = batch_ques.repeat_interleave(NUM_WALKS)
        next_k, q_weights = QK_Walker.sample_next(start_nodes, prev_weights=None)
        end_q, _ = KQ_Walker.sample_next(next_k, prev_weights=q_weights)

        paths = torch.stack([start_nodes, next_k, end_q], dim=1)
        all_paths.append(paths.cpu())
        
        torch.cuda.empty_cache()
    
    paths = torch.cat(all_paths, dim=0)
    print(f"\nSaving {paths.shape[0]} instances to {SAVE_PATH}...")
    np.save(SAVE_PATH, paths.numpy()) # 保存为 npy
    print("Done!")

if __name__ == "__main__":
    generate_qkq_instances()
