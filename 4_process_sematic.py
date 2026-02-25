import numpy as np
import pandas as pd
import os
import pickle

# ================= 配置区域 =================
BASE_DIR = os.getenv("MAGNN_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "2009")
MAPPED_DIR = os.path.join(BASE_DIR, "mapped_data")
OUT_DIR = os.path.join(MAPPED_DIR, "processed")
MAP_DIR = os.path.join(MAPPED_DIR, "maps")

# 输入文件路径
FILE_QUES_SKILL = os.path.join(MAPPED_DIR, "ques_skill_mapped.csv")
FILE_QUES_DISC = os.path.join(MAPPED_DIR, "ques_discvalue_mapped.csv")
# ===========================================

def load_pkl(name):
    path = os.path.join(MAP_DIR, f"{name}2idx.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)

def process_features_semantic():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    print("正在生成语义特征 (Semantic Features)...")

    # 1. 准备基础信息
    # 我们需要知道总共有多少个 Skill，这决定了特征向量的基础长度
    skill_map = load_pkl("skill")
    num_skills = len(skill_map)
    print(f"检测到知识点数量 (Num_Skills): {num_skills}")
    
    # 设定统一的特征维度 = Skill数量 + 1 (难度位)
    # 这样既能放下语义信息，又能保证维度对齐
    FEAT_DIM = num_skills + 1
    print(f"设定统一特征维度 (FEAT_DIM): {FEAT_DIM} (Skills + Difficulty)")

    # ================= 1. 问题 (Question) 特征 =================
    # 组成: [Skill_Multi_hot (N位) || Difficulty (1位)]
    print("\n[1/4] 生成问题特征 (Semantic)...")
    ques_map = load_pkl("ques")
    num_ques = len(ques_map)
    
    # 初始化为 0
    feat_q = np.zeros((num_ques, FEAT_DIM), dtype=np.float32)
    
    # A. 填充 Skill Multi-hot
    if os.path.exists(FILE_QUES_SKILL):
        df_qs = pd.read_csv(FILE_QUES_SKILL)
        # 确保是 int 类型
        if 'problem_id' in df_qs.columns:
            q_ids = df_qs['problem_id'].values.astype(int)
            s_ids = df_qs['skill_id'].values.astype(int)
        else:
            q_ids = df_qs['ques'].values.astype(int)
            s_ids = df_qs['skill'].values.astype(int)
        
        # 向量化赋值: 在对应的 (q, s) 位置设为 1
        # 注意边界检查，防止 csv 里有越界的 ID
        valid_mask = (q_ids < num_ques) & (s_ids < num_skills)
        feat_q[q_ids[valid_mask], s_ids[valid_mask]] = 1.0
    else:
        print(f"Warning: {FILE_QUES_SKILL} 不存在，问题特征将缺失技能信息！")

    # B. 填充 Difficulty
    if os.path.exists(FILE_QUES_DISC):
        df_disc = pd.read_csv(FILE_QUES_DISC)
        # CSV 列名是 'problem_id' 和 'discrimination'
        # 归一化难度值到 0~1 (如果原本不是的话)
        vals = df_disc['discrimination'].values.astype(np.float32)
        # Min-Max 归一化 (防守性编程)
        if vals.max() > 1.0 or vals.min() < 0.0:
            vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
            
        if 'problem_id' in df_disc.columns:
            q_ids = df_disc['problem_id'].values.astype(int)
        else:
            q_ids = df_disc['ques_id'].values.astype(int)
        valid_mask = q_ids < num_ques
        
        # 填入最后一位
        feat_q[q_ids[valid_mask], -1] = vals[valid_mask]
    else:
        print(f"Warning: {FILE_QUES_DISC} 不存在，问题特征将缺失难度信息！")

    np.save(os.path.join(OUT_DIR, "features_0.npy"), feat_q)
    print(f"  -> Saved Question Feature: {feat_q.shape}")

    # ================= 2. 学生 (User) 特征 =================
    # 策略: 随机初始化 (Cold Start)
    # 维度必须也是 FEAT_DIM
    print("\n[2/4] 生成学生特征 (Random)...")
    user_map = load_pkl("user")
    num_users = len(user_map)
    
    # 使用标准正态分布，并除以 sqrt(dim) 保持方差稳定，或者直接 randn
    feat_u = np.random.randn(num_users, FEAT_DIM).astype(np.float32)
    np.save(os.path.join(OUT_DIR, "features_1.npy"), feat_u)
    print(f"  -> Saved User Feature: {feat_u.shape}")

    # ================= 3. 知识点 (Skill) 特征 =================
    # 策略: One-hot Identity
    # 第 i 个 Skill 的特征应该是: 第 i 位为 1，其他为 0 (类似 Identity Matrix)
    # 最后一位(难度位) 补 0
    print("\n[3/4] 生成知识点特征 (One-hot Identity)...")
    
    # 创建一个单位矩阵 [num_skills, num_skills]
    identity = np.eye(num_skills, dtype=np.float32)
    # 补充最后的一列 0 (为了匹配难度位) -> [num_skills, 1]
    padding = np.zeros((num_skills, 1), dtype=np.float32)
    
    feat_s = np.concatenate([identity, padding], axis=1) # [num_skills, FEAT_DIM]
    
    np.save(os.path.join(OUT_DIR, "features_2.npy"), feat_s)
    print(f"  -> Saved Skill Feature: {feat_s.shape}")

    # ================= 4. 群组 (Cluster) 特征 =================
    # 策略: Random (因为 Cluster 数量远小于 FEAT_DIM，One-hot 会导致后面全是 0)
    # 或者: 如果 Cluster 数量很少 (8个)，用 One-hot 填充前 8 位也可以
    # 这里建议: 随机 (Random) 比较通用，因为 Cluster 只是隐语义
    print("\n[4/4] 生成群组特征 (Random)...")
    # 假设 Cluster 数量固定，或者你可以读取 cluster 文件
    # 这里为了简单，读取之前生成的 features_3 看看大小，或者假设是 8
    num_clusters = 8 # 你的代码里写的是 8
    
    feat_c = np.random.randn(num_clusters, FEAT_DIM).astype(np.float32)
    np.save(os.path.join(OUT_DIR, "features_3.npy"), feat_c)
    print(f"  -> Saved Cluster Feature: {feat_c.shape}")

    print("\n语义特征生成完毕！")

if __name__ == "__main__":
    process_features_semantic()
