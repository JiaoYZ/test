import pandas as pd
import numpy as np
import os

# ================= 🔧 配置区域 =================

# 训练集数据路径 (用于构建 ques_skill 图结构和计算难度属性)
TRAIN_PATHS = {
    # "2009": r"D:\Study\Study_Paper\code\data_process\2009_final_split\train_for_graph.csv",
    # "2012": r"D:\Study\Study_Paper\code\data_process\2012_final_split\train_for_graph.csv",
    # "2017": r"D:\Study\Study_Paper\code\data_process\2017_final_split\train_for_graph.csv",
    "ednet": r"D:\Study\Study_Paper\code\data_process\ednet_final_split\train_for_graph.csv",
}

# 输出目录建议也改一下，保持项目整洁
DATA_ROOT = r"D:\Study\Study_Paper\code\data_process\graph_data"

# ================= 🛠️ 核心处理逻辑 =================

def ensure_dirs(dataset_name):
    """创建 graph 和 attribute 文件夹"""
    paths = {
        "graph": os.path.join(DATA_ROOT, dataset_name, "graph"),
        "attribute": os.path.join(DATA_ROOT, dataset_name, "attribute")
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def process_dataset(name):
    print(f"\n🚀 正在处理数据集: {name} ...")
    
    train_path = TRAIN_PATHS.get(name)
    
    if not os.path.exists(train_path):
        print(f"❌ 文件缺失，跳过: {train_path}")
        return

    paths = ensure_dirs(name)

    # ---------------------------------------------------------
    # 第一步：构建 Graph (ques_skill.csv) - 使用训练集数据
    # ---------------------------------------------------------
    print("   [1/3] 构建训练集 ques_skill 图...")
    # 读取训练集数据
    # 指定 dtype 防止 skill_id 被误判 (比如 '1;13')
    df_train = pd.read_csv(train_path, dtype={'skill_id': str, 'problem_id': int})
    
    # 提取 problem_id 和 skill_id
    qs_temp = df_train[['problem_id', 'skill_id']].drop_duplicates().copy()
    
    # === 关键修改点开始 ===
    # 1. 先将字符串分割成列表，生成临时列 'skill_list'
    qs_temp['skill_list'] = qs_temp['skill_id'].astype(str).str.split(';')
    
    # 2. 对整个 DataFrame 进行 explode，这样 problem_id 会自动复制
    qs_exploded = qs_temp.explode('skill_list')
    
    # 3. 提取需要的列并重命名
    qs_clean = qs_exploded[['problem_id', 'skill_list']].copy()
    qs_clean.columns = ['ques', 'skill']
    # === 关键修改点结束 ===

    # 清洗：去除非数字的技能ID
    qs_clean = qs_clean[qs_clean['skill'].str.isnumeric()] # 确保是数字
    qs_clean['skill'] = qs_clean['skill'].astype(int)
    qs_clean['ques'] = qs_clean['ques'].astype(int)
    
    # 去重并保存
    qs_final = qs_clean.drop_duplicates()
    qs_final.to_csv(os.path.join(paths['graph'], "ques_skill.csv"), index=False)
    print(f"         已保存 ques_skill.csv (行数: {len(qs_final)})")

    # ---------------------------------------------------------
    # 第二步：计算属性 (Difficulty) - 使用训练集
    # ---------------------------------------------------------
    print("   [2/3] 计算训练集难度 (Difficulty)...")
    
    # 确保列名匹配 (兼容 user_id/problem_id 或 stu/ques)
    if 'ques' not in df_train.columns and 'problem_id' in df_train.columns:
        df_train.rename(columns={'problem_id': 'ques'}, inplace=True)
    
    # 计算难度: Diff = 1 - Mean(Correct)
    # GroupBy ques
    ques_stats = df_train.groupby('ques')['correct'].agg(['mean', 'count'])
    ques_stats['difficulty'] = 1 - ques_stats['mean']
    
    # 获取训练集计算出的难度字典
    train_diff_dict = ques_stats['difficulty'].to_dict()
    
    # ---------------------------------------------------------
    # 第三步：保存字典
    # ---------------------------------------------------------
    print("   [3/3] 保存难度字典并生成 Skill Variance...")
    
    # 直接使用训练集中计算出的难度字典，不需要填充
    final_diff_dict = train_diff_dict
            
    # 保存难度字典
    with open(os.path.join(paths['attribute'], "quesID2diffValue_dict.txt"), 'w') as f:
        f.write(str(final_diff_dict))
    print(f"         已保存 quesID2diffValue_dict.txt (题目数: {len(final_diff_dict)})")
    
    # ---------------------------------------------------------
    # 附加步：生成 skill_var.csv (SimKT 公式必须)
    # ---------------------------------------------------------
    
    # 1. 建立 skill -> [diff1, diff2...] 列表
    skill_diffs = {}
    
    # 为了加速，先转字典
    # qs_final 已经是炸裂后的 (ques, skill)
    q_to_s_list = qs_final.groupby('ques')['skill'].apply(list).to_dict()
    
    for qid, diff in final_diff_dict.items():
        if qid in q_to_s_list:
            skills = q_to_s_list[qid]
            for s in skills:
                if s not in skill_diffs: skill_diffs[s] = []
                skill_diffs[s].append(diff)
    
    # 2. 计算方差
    skill_var_list = []
    # 确保技能ID排序
    all_skills = sorted(qs_final['skill'].unique())
    
    for s in all_skills:
        diffs = skill_diffs.get(s, [])
        if len(diffs) > 1:
            var = np.var(diffs)
        else:
            var = 0.0 # 只有一个题或无题
        skill_var_list.append({'skill': s, 'variance': var})
        
    pd.DataFrame(skill_var_list).to_csv(os.path.join(paths['graph'], "skill_var.csv"), index=False)
    print(f"         已保存 skill_var.csv (用于 SimKT 公式计算)")

    print("-" * 50)

# ================= ▶️ 主程序 =================
if __name__ == "__main__":
    for name in TRAIN_PATHS.keys():
        process_dataset(name)
    print("\n✅ 所有数据集处理完毕！")