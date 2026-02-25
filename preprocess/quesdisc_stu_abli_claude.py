import pandas as pd
import numpy as np
import os

TRAIN_PATHS = {
    # "2009": r"D:\Study\Study_Paper\code\data_process\2009_final_split\train_for_graph.csv",
    # "2012": r"D:\Study\Study_Paper\code\data_process\2012_final_split\train_for_graph.csv",
    # "2017": r"D:\Study\Study_Paper\code\data_process\2017_final_split\train_for_graph.csv",
    "ednet": r"D:\Study\Study_Paper\code\data_process\ednet_final_split\train_for_graph.csv",
}

# 输出目录建议也改一下，保持项目整洁
DATA_ROOT = r"D:\Study\Study_Paper\code\data_process\graph_data"

def generate_ability_and_discrimination(name):
    """
    完全符合SimQE论文的学生能力和题目区分度计算
    """
    print(f"\n🚀 生成学生能力和题目区分度 (SimQE Paper): {name}")
    
    train_path = TRAIN_PATHS[name]
    qs_path = os.path.join(DATA_ROOT, name, "graph", "ques_skill.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(qs_path):
        print(f"❌ 缺少文件: {train_path} 或 {qs_path}")
        return

    # ===== 1. 读取数据 =====
    df_train = pd.read_csv(train_path)
    df_qs = pd.read_csv(qs_path)
    merged_df = pd.merge(df_train[['user_id', 'problem_id', 'correct']], df_qs, left_on='problem_id', right_on='ques', how='inner')
    
    print(f"   📊 训练数据: {len(df_train)} 条")
    print(f"   📊 关联技能后: {len(merged_df)} 条")
    
    # ========================================================================
    # Part A: 计算学生能力 (Definition 7)
    # ========================================================================
    print("\n   === Part A: 学生能力 ===")
    
    # --- Step A1: 计算题目难度 (Definition 4) ---
    print("   [1/4] 计算题目难度 D_q (Definition 4)...")
    ques_stats = df_train.groupby('problem_id').agg(
        correct_count=('correct', 'sum'),
        total_count=('correct', 'count')
    )
    # D_q = c_q / n_q (正确人数 / 总人数)
    ques_stats['difficulty'] = ques_stats['correct_count'] / ques_stats['total_count']
    ques_diff_dict = ques_stats['difficulty'].to_dict()
    
    # --- Step A2: 计算技能难度 D_k ---
    print("   [2/4] 计算技能难度 D_k (题目难度的平均)...")
    # 技能难度 = 该技能下所有题目的平均难度
    df_qs_with_diff = pd.merge(df_qs, ques_stats[['difficulty']], left_on='ques', right_index=True)
    skill_diff_dict = df_qs_with_diff.groupby('skill')['difficulty'].mean().to_dict()
    
    print(f"      技能难度范围: [{min(skill_diff_dict.values()):.3f}, {max(skill_diff_dict.values()):.3f}]")
    
    # --- Step A3: 计算学生在每个技能上的表现 (C_u^k - W_u^k) ---
    print("   [3/4] 计算学生-技能表现 (C_u^k - W_u^k)...")
    stu_skill_stats = merged_df.groupby(['user_id', 'skill'])['correct'].agg(['mean', 'count']).reset_index()
    stu_skill_stats.rename(columns={'mean': 'C_u_k'}, inplace=True)
    
    # W_u_k = 1 - C_u_k
    # (C - W) = 2*C - 1
    stu_skill_stats['perf_term'] = 2 * stu_skill_stats['C_u_k'] - 1
    
    # --- Step A4: 应用公式计算能力 ---
    print("   [4/4] 应用 Definition 7 公式...")
    stu_skill_stats['D_k'] = stu_skill_stats['skill'].map(skill_diff_dict)
    stu_skill_stats['weighted_score'] = stu_skill_stats['perf_term'] * stu_skill_stats['D_k']
    
    # 修正: 使用每个学生实际接触的技能数作为分母
    stu_ability_df = stu_skill_stats.groupby('user_id').agg({
        'weighted_score': 'sum',
        'skill': 'count'  # 学生接触的技能数
    }).reset_index()
    stu_ability_df.rename(columns={'skill': 'n_k'}, inplace=True)
    
    # B_u = Sum[(C - W) * D_k] / n_k
    stu_ability_df['ability_raw'] = stu_ability_df['weighted_score'] / stu_ability_df['n_k']
    
    # 归一化到 [0, 1]
    # 原始值在 [-1, 1] 之间
    stu_ability_df['ability'] = (stu_ability_df['ability_raw'] + 1) / 2
    
    ability_dict = stu_ability_df.set_index('user_id')['ability'].to_dict()
    
    print(f"      学生能力范围: [{min(ability_dict.values()):.3f}, {max(ability_dict.values()):.3f}]")
    print(f"      学生能力均值: {np.mean(list(ability_dict.values())):.3f}")
    print(f"      学生能力标准差: {np.std(list(ability_dict.values())):.3f}")
    
    # ========================================================================
    # Part B: 计算题目区分度 (Definition 6)
    # ========================================================================
    print("\n   === Part B: 题目区分度 ===")
    
    # --- Step B1: 根据能力给学生排序 ---
    print("   [1/3] 根据能力划分高/低能力学生...")
    df_train_with_abi = df_train.copy()
    df_train_with_abi['stu_ability'] = df_train_with_abi['user_id'].map(ability_dict)
    
    # 对于没有能力值的学生,用中位数填充
    median_ability = np.median(list(ability_dict.values()))
    df_train_with_abi['stu_ability'].fillna(median_ability, inplace=True)
    
    # 划分 top 50% 和 bottom 50%
    ability_threshold = df_train_with_abi['stu_ability'].median()
    df_train_with_abi['is_high_ability'] = df_train_with_abi['stu_ability'] >= ability_threshold
    
    # --- Step B2: 分别计算高/低能力学生在每个题目上的难度 ---
    print("   [2/3] 计算 D_q^H 和 D_q^L...")
    
    # D_q^H: 高能力学生在题q上的难度
    high_abi_stats = df_train_with_abi[df_train_with_abi['is_high_ability']].groupby('problem_id')['correct'].agg(['sum', 'count'])
    high_abi_stats['D_H'] = high_abi_stats['sum'] / high_abi_stats['count']
    
    # D_q^L: 低能力学生在题q上的难度
    low_abi_stats = df_train_with_abi[~df_train_with_abi['is_high_ability']].groupby('problem_id')['correct'].agg(['sum', 'count'])
    low_abi_stats['D_L'] = low_abi_stats['sum'] / low_abi_stats['count']
    
    # --- Step B3: 计算区分度 D̂_q = D_H - D_L ---
    print("   [3/3] 计算区分度 D̂_q = D_H - D_L...")
    disc_df = pd.DataFrame({
        'D_H': high_abi_stats['D_H'],
        'D_L': low_abi_stats['D_L']
    })
    disc_df['discrimination'] = disc_df['D_H'] - disc_df['D_L']
    
    # 缺失值填充(某些题目可能只有高能力或低能力学生做过)
    disc_df['discrimination'].fillna(0, inplace=True)
    
    disc_dict = disc_df['discrimination'].to_dict()
    
    print(f"      区分度范围: [{min(disc_dict.values()):.3f}, {max(disc_dict.values()):.3f}]")
    print(f"      区分度均值: {np.mean(list(disc_dict.values())):.3f}")
    print(f"      区分度标准差: {np.std(list(disc_dict.values())):.3f}")
    
    # ========================================================================
    # Part C: 保存文件
    # ========================================================================
    print("\n   === Part C: 保存文件 ===")
    
    # --- 直接使用训练集中的学生能力和题目区分度 ---
    final_ability_dict = {uid: float(ability_dict[uid]) for uid in ability_dict}
    final_disc_dict = {qid: float(disc_dict[qid]) for qid in disc_dict}
    
    # --- 保存文件 ---
    save_dir = os.path.join(DATA_ROOT, name, "graph")
    os.makedirs(save_dir, exist_ok=True)
    
    # 学生能力
    ability_df = pd.DataFrame(list(final_ability_dict.items()), columns=['stu_id', 'ability'])
    ability_path = os.path.join(save_dir, "stu_abi.csv")
    ability_df.to_csv(ability_path, index=False)
    print(f"   ✅ 学生能力: {ability_path} (学生数: {len(final_ability_dict)})")
    
    # 题目区分度
    disc_df = pd.DataFrame(list(final_disc_dict.items()), columns=['ques_id', 'discrimination'])
    disc_path = os.path.join(save_dir, "ques_discvalue.csv")
    disc_df.to_csv(disc_path, index=False)
    print(f"   ✅ 题目区分度: {disc_path} (题目数: {len(final_disc_dict)})")
    
    # ========================================================================
    # Part D: 统计分析
    # ========================================================================
    print("\n   === 统计摘要 ===")
    print(f"   学生能力分布:")
    print(f"      - 均值: {np.mean(list(ability_dict.values())):.4f}")
    print(f"      - 标准差: {np.std(list(ability_dict.values())):.4f}")
    print(f"      - 中位数: {np.median(list(ability_dict.values())):.4f}")
    print(f"      - 范围: [{min(ability_dict.values()):.4f}, {max(ability_dict.values()):.4f}]")
    
    print(f"\n   题目区分度分布:")
    print(f"      - 均值: {np.mean(list(disc_dict.values())):.4f}")
    print(f"      - 标准差: {np.std(list(disc_dict.values())):.4f}")
    print(f"      - 中位数: {np.median(list(disc_dict.values())):.4f}")
    print(f"      - 范围: [{min(disc_dict.values()):.4f}, {max(disc_dict.values()):.4f}]")


if __name__ == "__main__":
    for name in TRAIN_PATHS.keys():
        generate_ability_and_discrimination(name)