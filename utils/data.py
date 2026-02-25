import torch
import numpy as np
import pickle
import os
import pandas as pd
from collections import defaultdict

# ================= 配置路径 =================
def resolve_base_dir(base_dir=None, dataset=None):
    if base_dir:
        return os.path.abspath(base_dir)
    env_dir = os.getenv("MAGNN_DATA_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    if dataset:
        return os.path.abspath(os.path.join(here, "..", dataset))
    return os.path.abspath(os.path.join(here, "..", "2017"))

def get_paths(base_dir=None, dataset=None):
    base_dir = resolve_base_dir(base_dir, dataset)
    mapped_dir = os.path.join(base_dir, "mapped_data")
    processed_dir = os.path.join(mapped_dir, "processed")
    return base_dir, mapped_dir, processed_dir

def convert_paths_to_idx_list(paths, num_questions, path_len):
    """
    将 DGL 采样的路径转换为 MAGNN 需要的 idx_lists 格式
    
    Args:
        paths: list of tuples, e.g. [(q1, u1, q1), (q2, u2, q2)...]
        num_questions: 题目总数，用于初始化列表大小
        path_len: 路径长度 (不包含起点), e.g. Q-U-Q 是 2
    
    Returns:
        idx_list: list of numpy arrays. idx_list[q_id] = [[u1, q1], [u2, q2]...]
    """
    # 1. 初始化字典
    # 使用 list 而不是 set，因为我们已经在采样阶段去重了
    adj_dict = defaultdict(list)
    
    # 2. 填充数据
    for path in paths:
        # path: (start, n1, n2, ...)
        start_node = path[0]
        # 截取后面的节点作为邻居路径 (去掉起点)
        neighbor_path = list(path[1:])
        
        # 确保长度一致 (为了安全)
        if len(neighbor_path) == path_len:
            adj_dict[start_node].append(neighbor_path)
            
    # 3. 转换为 list of arrays (MAGNN 格式)
    idx_list = []
    for q_id in range(num_questions):
        if q_id in adj_dict:
            # 转换为 numpy 数组
            idx_list.append(np.array(adj_dict[q_id], dtype=np.int64))
        else:
            # 如果是孤立点，或者没采到，填一个空的 array
            # 注意形状要对齐，虽然是空的
            idx_list.append(np.zeros((0, path_len), dtype=np.int64))
            
    return idx_list

def get_batch(sequences, batch_size, max_seq_len):
    """
    生成 Batch 数据
    输入:
        sequences: list of (q_seq, a_seq)
        batch_size: 批次大小
        max_seq_len: 最大序列长度
    输出:
        yield (q_batch, a_batch, mask_batch) 均为 Tensor 且已移动到 GPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_len = len(sequences)
    indices = np.arange(data_len)
    np.random.shuffle(indices) # 每个 Epoch 打乱数据
    
    for i in range(0, data_len, batch_size):
        batch_indices = indices[i : i + batch_size]
        
        q_batch = []
        a_batch = []
        mask_batch = []
        
        for idx in batch_indices:
            q_seq, a_seq = sequences[idx]
            
            # 1. 截断 (从后往前截取最新的交互)
            if len(q_seq) > max_seq_len:
                q_seq = q_seq[-max_seq_len:]
                a_seq = a_seq[-max_seq_len:]
            
            # 2. 填充 (Padding)
            pad_len = max_seq_len - len(q_seq)
            
            # 这里的填充值通常用 0，模型中需要配合 mask 使用
            padded_q = q_seq + [0] * pad_len
            padded_a = a_seq + [0] * pad_len
            mask = [1] * len(q_seq) + [0] * pad_len
            
            q_batch.append(padded_q)
            a_batch.append(padded_a)
            mask_batch.append(mask)
            
        # 转换并移动到设备
        yield (torch.LongTensor(q_batch).to(device), 
               torch.LongTensor(a_batch).to(device), 
               torch.FloatTensor(mask_batch).to(device))

def load_data(base_dir=None, dataset=None):
    """
    加载 MAGNN 需要的图结构数据和特征矩阵
    返回: (adjlists, idx_lists, features_list)
    """
    _, _, processed_dir = get_paths(base_dir, dataset)
    print(f"正在从 {processed_dir} 加载图数据...")
    
    # 1. 定义元路径顺序 (必须与 run_KT.py 中的 metapath_list 一致)
    # 0:Question, 1:User, 2:Skill, 3:Cluster
    mp_names = ["0-1-0", "0-2-0", "0-1-3-1-0"] 
    
    # 2. 加载邻接表 (Adjacency Lists)
    # 对应 processed/0/xxx.adjlist
    adjlists = []
    for mp in mp_names:
        filename = os.path.join(processed_dir, "0", f"{mp}.adjlist")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"找不到文件: {filename}。请检查是否运行了 3_convert_to_magnn.py")
            
        adj = []
        with open(filename, 'r') as f:
            for line in f:
                # 文件格式: [src_id] [neighbor_1] [neighbor_2] ...
                parts = list(map(int, line.strip().split()))
                # 我们只需要邻居列表 (parts[1:])，不需要 src_id (parts[0])
                # 因为 adj[i] 就代表节点 i 的邻居
                adj.append(parts[1:]) 
        adjlists.append(adj)

    # 3. 加载路径索引 (Pickle Files)
    # 对应 processed/0/xxx_idx.pickle
    idx_lists = []
    for mp in mp_names:
        filename = os.path.join(processed_dir, "0", f"{mp}_idx.pickle")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"找不到文件: {filename}")
            
        with open(filename, 'rb') as f:
            idx = pickle.load(f)
        idx_lists.append(idx)

    # 4. 加载特征矩阵 (Features)
    # 对应 processed/features_0.npy 等
    features_list = []
    # 我们有 4 种节点类型: 0, 1, 2, 3
    for i in range(4): 
        filename = os.path.join(processed_dir, f"features_{i}.npy")
        if os.path.exists(filename):
            feat = np.load(filename)
            features_list.append(feat)
        else:
            print(f"Warning: features_{i}.npy 不存在，该类型节点将没有预训练特征。")
            features_list.append(None)

    print("图数据加载完成。")
    return adjlists, idx_lists, features_list

def load_kt_sequences(base_dir=None, dataset=None):
    """
    加载 train_mapped.txt 和 test_mapped.txt
    返回: train_seqs, test_seqs
    格式: List of (question_sequence, answer_sequence)
    """
    _, mapped_dir, _ = get_paths(base_dir, dataset)
    print(f"正在从 {mapped_dir} 加载序列数据...")
    
    def parse_txt(filepath):
        sequences = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到序列文件: {filepath}。请检查 0_remap_everything.py 是否成功运行。")
            
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # 文件格式是 4 行一组:
            # Line 1: 序列长度
            # Line 2: 题目 ID 序列 (逗号分隔)
            # Line 3: 知识点 ID 序列
            # Line 4: 对错结果序列
            
            i = 0
            while i < len(lines):
                if len(lines[i].strip()) == 0: 
                    i += 1
                    continue
                    
                # 读取题目序列 (Line 2)
                q_seq = list(map(int, lines[i+1].strip().split(',')))
                
                # 读取对错序列 (Line 4)
                a_seq = list(map(int, lines[i+3].strip().split(',')))
                
                # 简单的完整性检查
                if len(q_seq) != len(a_seq):
                    print(f"Warning: Line {i} 题目和答案长度不匹配，跳过。")
                else:
                    sequences.append((q_seq, a_seq))
                
                i += 4 # 跳到下一组
        return sequences

    train_path = os.path.join(mapped_dir, "train_mapped.txt")
    test_path = os.path.join(mapped_dir, "test_mapped.txt")
    
    train_seqs = parse_txt(train_path)
    test_seqs = parse_txt(test_path)
    
    print(f"序列加载完成: Train={len(train_seqs)}, Test={len(test_seqs)}")
    return train_seqs, test_seqs

def load_question_skill_map(base_dir=None, dataset=None):
    base_dir, mapped_dir, _ = get_paths(base_dir, dataset)
    qs_path = os.path.join(mapped_dir, "ques_skill_mapped.csv")
    if not os.path.exists(qs_path):
        raise FileNotFoundError(f"找不到文件: {qs_path}")
    df = pd.read_csv(qs_path)
    if 'ques' in df.columns and 'skill' in df.columns:
        q_col, s_col = 'ques', 'skill'
    elif 'problem_id' in df.columns and 'skill_id' in df.columns:
        q_col, s_col = 'problem_id', 'skill_id'
    else:
        raise ValueError(f"Unknown column names in ques_skill_mapped.csv: {df.columns.tolist()}")

    df[q_col] = pd.to_numeric(df[q_col], errors='coerce')
    df[s_col] = pd.to_numeric(df[s_col], errors='coerce')
    df = df.dropna().astype(int)

    ques_map_path = os.path.join(mapped_dir, "maps", "ques2idx.pkl")
    skill_map_path = os.path.join(mapped_dir, "maps", "skill2idx.pkl")
    if os.path.exists(ques_map_path):
        with open(ques_map_path, 'rb') as f:
            ques2idx = pickle.load(f)
        num_questions = len(ques2idx)
    else:
        num_questions = int(df[q_col].max()) + 1
    if os.path.exists(skill_map_path):
        with open(skill_map_path, 'rb') as f:
            skill2idx = pickle.load(f)
        num_skills = len(skill2idx)
    else:
        num_skills = int(df[s_col].max()) + 1

    skills_per_q = [[] for _ in range(num_questions)]
    for q, s in zip(df[q_col].values, df[s_col].values):
        if 0 <= q < num_questions:
            skills_per_q[q].append(int(s))

    dummy_skill = num_skills
    has_empty = False
    for i in range(num_questions):
        if len(skills_per_q[i]) == 0:
            skills_per_q[i] = [dummy_skill]
            has_empty = True
        else:
            skills_per_q[i] = list(set(skills_per_q[i]))

    if has_empty:
        num_skills = num_skills + 1

    skill_indices = []
    skill_offsets = [0]
    for skills in skills_per_q:
        skill_indices.extend(skills)
        skill_offsets.append(skill_offsets[-1] + len(skills))

    skill_indices = torch.LongTensor(skill_indices)
    skill_offsets = torch.LongTensor(skill_offsets)
    return skill_indices, skill_offsets, num_skills, num_questions

def load_transition_adj(num_questions, base_dir=None, dataset=None, device=None):
    import scipy.sparse as sp
    base_dir, mapped_dir, _ = get_paths(base_dir, dataset)
    train_path = os.path.join(mapped_dir, "train_mapped.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到序列文件: {train_path}")

    def iterate_sequences(filepath):
        with open(filepath, 'r') as f:
            while True:
                len_line = f.readline()
                if not len_line:
                    break
                if len(len_line.strip()) == 0:
                    continue
                q_line = f.readline()
                _ = f.readline()
                a_line = f.readline()
                if not q_line or not a_line:
                    break
                q_seq = list(map(int, q_line.strip().split(',')))
                a_seq = list(map(int, a_line.strip().split(',')))
                if len(q_seq) == len(a_seq) and len(q_seq) > 1:
                    yield q_seq, a_seq

    total_edges = 0
    for q_seq, a_seq in iterate_sequences(train_path):
        total_edges += len(q_seq) - 1

    rows = np.empty(total_edges, dtype=np.int32)
    cols = np.empty(total_edges, dtype=np.int32)
    idx = 0
    for q_seq, a_seq in iterate_sequences(train_path):
        for i in range(len(q_seq) - 1):
            curr = q_seq[i] + (0 if a_seq[i] == 1 else num_questions)
            nxt = q_seq[i + 1] + (0 if a_seq[i + 1] == 1 else num_questions)
            rows[idx] = curr
            cols[idx] = nxt
            idx += 1
    data = np.ones(total_edges, dtype=np.float32)
    size = 2 * num_questions
    resout = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    resin = resout.T.tocsr()

    def normalize(mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx)

    resout = normalize(resout + sp.eye(resout.shape[0]))
    resin = normalize(resin + sp.eye(resin.shape[0]))

    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    resout = sparse_mx_to_torch_sparse_tensor(resout)
    resin = sparse_mx_to_torch_sparse_tensor(resin)
    if device is not None:
        resout = resout.to(device)
        resin = resin.to(device)
    return resout, resin

if __name__ == "__main__":
    # 简单的测试代码，运行该脚本可检查加载是否正常
    try:
        adj, idx, feat = load_data()
        print("Graph Load Success!")
        tr, te = load_kt_sequences()
        print("Sequence Load Success!")
    except Exception as e:
        print(f"加载出错: {e}")
