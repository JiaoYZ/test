import numpy as np
import pickle
import os
from collections import defaultdict

# ================= 配置区域 =================
BASE_DIR = os.getenv("MAGNN_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "2009")
MAPPED_DIR = os.path.join(BASE_DIR, "mapped_data")
INSTANCE_DIR = os.path.join(MAPPED_DIR, "instances")
# 输出目录：放在 processed/0 下，代表以类型0(题目)为核心
OUT_DIR = os.path.join(MAPPED_DIR, "processed", "0") 

# 定义文件对应关系
# Key: 你的 .npy 文件名
# Value: MAGNN 的元路径代码 (0:Question, 1:User, 2:Skill, 3:Cluster)
FILES_TO_CONVERT = {
    "quq_paths_100.npy": "0-1-0",       # Q-U-Q
    "qkq_paths_100.npy": "0-2-0",       # Q-K-Q
    "qucuq_paths_100.npy": "0-1-3-1-0"  # Q-U-C-U-Q
}

# 加载映射表以确定题目总数（防止漏掉孤立点）
MAP_PATH = os.path.join(MAPPED_DIR, "maps", "ques2idx.pkl")
# ===========================================

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def load_num_ques():
    if not os.path.exists(MAP_PATH):
        print(f"错误：找不到映射表 {MAP_PATH}，请检查路径。")
        return 0
    with open(MAP_PATH, 'rb') as f:
        ques2idx = pickle.load(f)
    # 确保总数是 Max ID + 1
    return len(ques2idx)

def convert_file(filename, type_code, num_ques):
    npy_path = os.path.join(INSTANCE_DIR, filename)
    if not os.path.exists(npy_path):
        print(f"[跳过] 文件不存在: {npy_path}")
        return

    print(f"正在转换 {filename} -> {type_code} ...")
    
    # 1. 加载路径数据
    try:
        paths = np.load(npy_path) # [N_total, Path_Len]
    except Exception as e:
        print(f"读取错误: {e}")
        return

    # 2. 聚合数据
    # adj_dict: 存储每个题目对应的“终点”邻居列表
    # path_dict: 存储每个题目对应的“完整路径”列表
    adj_dict = defaultdict(list)
    path_dict = defaultdict(list)
    
    for p in paths:
        src = int(p[0])  # 起始节点 (Question)
        dst = int(p[-1]) # 终止节点 (Neighbor Question)
        adj_dict[src].append(dst)
        path_dict[src].append(p)
        
    # 3. 写入 .adjlist 文件
    # MAGNN 格式: [Node_ID] [Neighbor1] [Neighbor2] ...
    adj_file = os.path.join(OUT_DIR, f"{type_code}.adjlist")
    with open(adj_file, "w") as f:
        for qid in range(num_ques):
            neighbors = adj_dict[qid]
            # 即使没有邻居，也要写一行 "qid "，保持行号对应
            line_str = str(qid) + " " + " ".join(map(str, neighbors)) + "\n"
            f.write(line_str)
            
    # 4. 写入 _idx.pickle 文件
    # 格式: Numpy Object Array。第 i 个元素是节点 i 的路径矩阵。
    idx_file = os.path.join(OUT_DIR, f"{type_code}_idx.pickle")
    
    all_indices = []
    # 获取路径长度用于生成空占位符
    path_len = paths.shape[1]
    
    for qid in range(num_ques):
        if qid in path_dict:
            # 该题目有路径：直接转换
            p_arr = np.array(path_dict[qid], dtype=int)
            # 简单打乱顺序
            np.random.shuffle(p_arr)
            all_indices.append(p_arr)
        else:
            # 该题目是孤立点：填入一个空的数组
            all_indices.append(np.zeros((0, path_len), dtype=int))
            
    with open(idx_file, "wb") as f:
        pickle.dump(np.array(all_indices, dtype=object), f)
        
    print(f"完成! 生成了 {type_code}.adjlist 和 {type_code}_idx.pickle")

if __name__ == "__main__":
    num_nodes = load_num_ques()
    if num_nodes > 0:
        print(f"检测到题目总数: {num_nodes}")
        for fname, code in FILES_TO_CONVERT.items():
            convert_file(fname, code, num_nodes)
        print("\n所有路径转换完成！文件位于 mapped_data/processed/0/")
