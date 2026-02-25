import pandas as pd
import numpy as np
import os
import pickle
import ast
from tqdm import tqdm

# ================= 配置区域 =================
# 支持通过环境变量 MAGNN_DATA_DIR 指定数据目录，默认使用相对路径
BASE_DIR = os.getenv("MAGNN_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "2012")
GRAPH_DIR = os.path.join(BASE_DIR, "graph")
ATTR_DIR = os.path.join(BASE_DIR, "attribute")
SPLIT_DIR = os.path.join(BASE_DIR, "ednet_final_split")
if not os.path.exists(SPLIT_DIR):
    SPLIT_DIR = os.path.join(BASE_DIR, "2012_final_split")
if not os.path.exists(SPLIT_DIR):
    SPLIT_DIR = os.path.join(BASE_DIR, "2009_final_split")

# 输入文件列表
FILES = {
    'ques_disc': os.path.join(GRAPH_DIR, "ques_discvalue.csv"),
    'ques_skill': os.path.join(GRAPH_DIR, "ques_skill.csv"),
    'stu_abi': os.path.join(GRAPH_DIR, "stu_abi.csv"),
    'stu_cluster': os.path.join(GRAPH_DIR, "stu_cluster_8.csv"),
    'stu_ques': os.path.join(GRAPH_DIR, "stu_ques.csv"),
    'train_graph': os.path.join(SPLIT_DIR, "train_for_graph.csv"),
    
    'diff_dict': os.path.join(ATTR_DIR, "quesID2diffValue_dict.txt"),
    'train_txt': os.path.join(SPLIT_DIR, "train.txt"),
    'test_txt': os.path.join(SPLIT_DIR, "test.txt"),
}

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "mapped_data")
MAP_DIR = os.path.join(OUTPUT_DIR, "maps")
MIN_SEQ_LEN = 2
MAX_SEQ_LEN = 500

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(MAP_DIR): os.makedirs(MAP_DIR)

print(f"输入目录: {BASE_DIR}")
print(f"输出目录: {OUTPUT_DIR}")
print("-" * 30)

# ================= 第一步：建立全局 ID 映射 (Build Global Maps) =================
print("Step 1: 扫描所有文件，建立 ID 映射...")

all_ques = set()
all_users = set()
all_skills = set()

# 1.1 从 CSV 中收集 ID
print("  Scanning CSV files...")
# ques_skill.csv -> 根据数据集格式确定列名
df_qs = pd.read_csv(FILES['ques_skill'])
if 'ques' in df_qs.columns and 'skill' in df_qs.columns:
    # 2017 数据集格式
    all_ques.update(set(df_qs['ques'].dropna().unique()))
    all_skills.update(set(df_qs['skill'].dropna().unique()))
elif 'problem_id' in df_qs.columns and 'skill_id' in df_qs.columns:
    # 2009 数据集格式
    all_ques.update(set(df_qs['problem_id'].dropna().unique()))
    all_skills.update(set(df_qs['skill_id'].dropna().unique()))
else:
    raise ValueError(f"Unknown column names in ques_skill.csv: {df_qs.columns.tolist()}")

# train_for_graph.csv -> user_id, problem_id, skill_id
df_tg = pd.read_csv(FILES['train_graph'])
all_ques.update(set(df_tg['problem_id'].unique()))
all_users.update(set(df_tg['user_id'].unique()))
# 兼容 train_for_graph 中可能存在的 skill_id 列
if 'skill_id' in df_tg.columns:
    # 确保只添加数值型数据，过滤掉 NaN 或其他非数值类型
    skill_ids = df_tg['skill_id'].dropna()
    # 转换为整数并添加到集合中
    all_skills.update(set([int(x) for x in skill_ids if pd.notna(x) and str(x).isdigit()]))

# stu_ques.csv -> user_id, problem_id
df_sq = pd.read_csv(FILES['stu_ques'])
all_ques.update(set(df_sq['problem_id'].dropna().unique()))
all_users.update(set(df_sq['user_id'].dropna().unique()))

# stu_cluster_8.csv -> 根据数据集格式确定列名
df_sc = pd.read_csv(FILES['stu_cluster'])
if 'stu' in df_sc.columns:
    # 2017 数据集格式
    all_users.update(set(df_sc['stu'].dropna().unique()))
elif 'user_id' in df_sc.columns:
    # 2009 数据集格式
    all_users.update(set(df_sc['user_id'].dropna().unique()))
else:
    raise ValueError(f"Unknown column names in stu_cluster.csv: {df_sc.columns.tolist()}")

# 1.2 从 TXT (train.txt/test.txt) 中收集 ID
# 注意：TXT 里可能包含 CSV 里没出现过的冷门题目或技能，必须扫描
def scan_txt_ids(filepath):
    q_set, s_set = set(), set()
    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if i + 3 >= len(lines):
                break
            # 格式：长度 -> 题目 -> 技能 -> 答案
            # q_line: 1,2,3
            q_line = lines[i+1].strip().split(',')
            # s_line: 10;11,12,13 (可能有分号)
            s_line = lines[i+2].strip().split(',')
            
            for q in q_line:
                if q: q_set.add(int(q))
            
            for s_group in s_line:
                if s_group:
                    # 处理 "103;114" 这种情况
                    single_skills = s_group.split(';')
                    for s in single_skills:
                        if s: s_set.add(int(s))
            i += 4 # 假设是 4 行格式
    return q_set, s_set

print("  Scanning TXT files (this may take a moment)...")
q_train, s_train = scan_txt_ids(FILES['train_txt'])
q_test, s_test = scan_txt_ids(FILES['test_txt'])

all_ques.update(q_train)
all_ques.update(q_test)
all_skills.update(s_train)
all_skills.update(s_test)

# 1.3 生成映射字典 (Raw -> New Index)
# 排序很重要，保证每次运行结果一致
ques2idx = {raw: i for i, raw in enumerate(sorted(all_ques))}
user2idx = {raw: i for i, raw in enumerate(sorted(all_users))}
skill2idx = {raw: i for i, raw in enumerate(sorted(all_skills))}

print(f"  [统计] 题目总数: {len(ques2idx)}")
print(f"  [统计] 学生总数: {len(user2idx)}")
print(f"  [统计] 技能总数: {len(skill2idx)}")

# 保存映射表
with open(os.path.join(MAP_DIR, 'ques2idx.pkl'), 'wb') as f: pickle.dump(ques2idx, f)
with open(os.path.join(MAP_DIR, 'user2idx.pkl'), 'wb') as f: pickle.dump(user2idx, f)
with open(os.path.join(MAP_DIR, 'skill2idx.pkl'), 'wb') as f: pickle.dump(skill2idx, f)
print("  映射表已保存至 mapped_data/maps/")


# ================= 第二步：重映射 CSV 文件 =================
print("\nStep 2: 重映射 CSV 文件...")

def safe_map(series, mapping, name):
    series = pd.to_numeric(series, errors='coerce')
    mapped = series.map(mapping)
    if mapped.isnull().any():
        print(f"  [警告] 在 {name} 中发现未知的 ID，将变为 NaN 并被丢弃。")
    return mapped

# 2.1 train_for_graph.csv
print("  Processing train_for_graph.csv...")
# 根据数据集格式使用相应的列名
df_tg['problem_id'] = safe_map(df_tg['problem_id'], ques2idx, 'train_graph problem_id')
df_tg['user_id'] = safe_map(df_tg['user_id'], user2idx, 'train_graph user_id')
if 'skill_id' in df_tg.columns:
    df_tg['skill_id'] = safe_map(df_tg['skill_id'], skill2idx, 'train_graph skill_id')
if 'correct' in df_tg.columns:
    df_tg = df_tg[df_tg['correct'].isin([0, 1])]
# 过滤掉可能的 NaN (虽然 Step 1 应该覆盖了所有，但以防万一)
df_tg = df_tg.dropna().astype(int)
df_tg = df_tg.drop_duplicates()
df_tg.to_csv(os.path.join(OUTPUT_DIR, 'train_for_graph_mapped.csv'), index=False)

# 2.2 ques_skill.csv
print("  Processing ques_skill.csv...")
# 根据数据集格式使用相应的列名
if 'ques' in df_qs.columns and 'skill' in df_qs.columns:
    # 2017 数据集格式
    df_qs['ques'] = safe_map(df_qs['ques'], ques2idx, 'ques_skill ques')
    df_qs['skill'] = safe_map(df_qs['skill'], skill2idx, 'ques_skill skill')
elif 'problem_id' in df_qs.columns and 'skill_id' in df_qs.columns:
    # 2009 数据集格式
    df_qs['problem_id'] = safe_map(df_qs['problem_id'], ques2idx, 'ques_skill problem_id')
    df_qs['skill_id'] = safe_map(df_qs['skill_id'], skill2idx, 'ques_skill skill_id')
df_qs = df_qs.dropna().astype(int)
df_qs = df_qs.drop_duplicates()
df_qs.to_csv(os.path.join(OUTPUT_DIR, 'ques_skill_mapped.csv'), index=False)

# 2.3 stu_ques.csv
print("  Processing stu_ques.csv...")
df_sq['user_id'] = safe_map(df_sq['user_id'], user2idx, 'stu_ques user_id')
df_sq['problem_id'] = safe_map(df_sq['problem_id'], ques2idx, 'stu_ques problem_id')
df_sq = df_sq.dropna().astype(int)
df_sq = df_sq.drop_duplicates()
df_sq.to_csv(os.path.join(OUTPUT_DIR, 'stu_ques_mapped.csv'), index=False)

# 2.4 stu_cluster_8.csv
print("  Processing stu_cluster_8.csv...")
# 根据数据集格式使用相应的列名
if 'stu' in df_sc.columns:
    # 2017 数据集格式
    df_sc['stu'] = safe_map(df_sc['stu'], user2idx, 'stu_cluster stu')
elif 'user_id' in df_sc.columns:
    # 2009 数据集格式
    df_sc['user_id'] = safe_map(df_sc['user_id'], user2idx, 'stu_cluster user_id')
# 注意：cluster 列通常是类别标签(0-7)，不需要映射！
df_sc = df_sc.dropna().astype(int)
df_sc = df_sc.drop_duplicates()
df_sc.to_csv(os.path.join(OUTPUT_DIR, 'stu_cluster_8_mapped.csv'), index=False)

# 2.5 ques_discvalue.csv (特殊处理：这是属性文件)
print("  Processing ques_discvalue.csv...")
df_qd = pd.read_csv(FILES['ques_disc'])
# 根据数据集格式使用相应的列名
if 'ques_id' in df_qd.columns:
    # 2017 数据集格式
    df_qd['ques_id'] = df_qd['ques_id'].map(ques2idx)
elif 'problem_id' in df_qd.columns:
    # 2009 数据集格式
    df_qd['problem_id'] = df_qd['problem_id'].map(ques2idx)
# 这里不能 dropna，因为可能有题目没有区分度值，但如果是 NaN，模型可能读不了
# 建议：去掉不在映射表里的行，保留在映射表里的
if 'ques_id' in df_qd.columns:
    df_qd = df_qd.dropna(subset=['ques_id'])
    df_qd['ques_id'] = df_qd['ques_id'].astype(int)
    # 按新 ID 排序，方便后续处理成矩阵
    df_qd = df_qd.sort_values('ques_id')
elif 'problem_id' in df_qd.columns:
    df_qd = df_qd.dropna(subset=['problem_id'])
    df_qd['problem_id'] = df_qd['problem_id'].astype(int)
    # 按新 ID 排序，方便后续处理成矩阵
    df_qd = df_qd.sort_values('problem_id')
df_qd.to_csv(os.path.join(OUTPUT_DIR, 'ques_discvalue_mapped.csv'), index=False)

# 2.6 stu_abi.csv (特殊处理：属性文件)
print("  Processing stu_abi.csv...")
df_abi = pd.read_csv(FILES['stu_abi'])
# 根据数据集格式使用相应的列名
if 'stu_id' in df_abi.columns:
    # 2017 数据集格式
    df_abi['stu_id'] = df_abi['stu_id'].map(user2idx)
    df_abi = df_abi.dropna(subset=['stu_id'])
    df_abi['stu_id'] = df_abi['stu_id'].astype(int)
    df_abi = df_abi.sort_values('stu_id')
elif 'user_id' in df_abi.columns:
    # 2009 数据集格式
    df_abi['user_id'] = df_abi['user_id'].map(user2idx)
    df_abi = df_abi.dropna(subset=['user_id'])
    df_abi['user_id'] = df_abi['user_id'].astype(int)
    df_abi = df_abi.sort_values('user_id')
df_abi.to_csv(os.path.join(OUTPUT_DIR, 'stu_abi_mapped.csv'), index=False)


# ================= 第三步：重映射 Attribute 字典文件 =================
print("\nStep 3: 重映射 quesID2diffValue_dict.txt...")

with open(FILES['diff_dict'], 'r') as f:
    content = f.read()
    # 这里的 content 是 "{123: 0.5, 456: 0.2}" 这样的字符串
    raw_dict = ast.literal_eval(content)

new_dict = {}
for k, v in raw_dict.items():
    if k in ques2idx:
        new_id = ques2idx[k]
        new_dict[new_id] = v

# 保存为文本
with open(os.path.join(OUTPUT_DIR, 'quesID2diffValue_dict_mapped.txt'), 'w') as f:
    f.write(str(new_dict))
print(f"  字典已重映射，原有 {len(raw_dict)} 项，现有 {len(new_dict)} 项。")


# ================= 第四步：重映射 Train/Test TXT 文件 =================
print("\nStep 4: 重映射 TXT 数据集文件...")

def remap_seq_file(infile, outfile):
    with open(infile, 'r') as f_in, open(outfile, 'w') as f_out:
        lines = f_in.readlines()
        i = 0
        total_records = 0
        kept_records = 0
        while i < len(lines):
            if i + 3 >= len(lines):
                break
            # 读取一组记录 (假设4行格式)
            q_line = lines[i+1].strip()
            s_line = lines[i+2].strip()
            a_line = lines[i+3] # 包含换行符
            
            q_list = q_line.split(',')
            s_groups = s_line.split(',')
            a_list = a_line.strip().split(',')
            n = min(len(q_list), len(s_groups), len(a_list))
            new_q_list = []
            new_s_groups = []
            new_a_list = []
            for idx in range(n):
                q_raw = q_list[idx].strip()
                a_raw = a_list[idx].strip()
                s_raw = s_groups[idx].strip() if idx < len(s_groups) else ""
                if len(q_raw) == 0 or len(a_raw) == 0:
                    continue
                try:
                    q_id = int(q_raw)
                    a_id = int(a_raw)
                except Exception:
                    continue
                if a_id not in (0, 1):
                    continue
                if q_id not in ques2idx:
                    continue
                sub_skills_mapped = []
                if len(s_raw) > 0:
                    for s in s_raw.split(';'):
                        if len(s) == 0:
                            continue
                        try:
                            s_id = int(s)
                        except Exception:
                            continue
                        if s_id in skill2idx:
                            sub_skills_mapped.append(str(skill2idx[s_id]))
                new_q_list.append(str(ques2idx[q_id]))
                new_s_groups.append(";".join(sub_skills_mapped))
                new_a_list.append(str(a_id))
            if len(new_q_list) < MIN_SEQ_LEN or len(new_q_list) > MAX_SEQ_LEN:
                i += 4
                total_records += 1
                continue
            f_out.write(str(len(new_q_list)) + "\n")
            f_out.write(",".join(new_q_list) + "\n")
            f_out.write(",".join(new_s_groups) + "\n")
            f_out.write(",".join(new_a_list) + "\n")
            
            i += 4
            total_records += 1
            kept_records += 1
            
    print(f"  已完成 {os.path.basename(infile)} -> {kept_records}/{total_records} 条记录")

remap_seq_file(FILES['train_txt'], os.path.join(OUTPUT_DIR, 'train_mapped.txt'))
remap_seq_file(FILES['test_txt'], os.path.join(OUTPUT_DIR, 'test_mapped.txt'))

print("\n" + "="*30)
print("全部完成！所有重映射后的文件都在 mapped_data 文件夹中。")
print("接下来请只使用 mapped_data 里的文件进行图构建和模型训练！")
