import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ================= 🔧 配置区域 =================
DATA_ROOT = r"D:\Study\Study_Paper\code\data_process"
# datasets = ["ASSIST09", "ASSIST12", "ASSIST17", "EdNet"]
datasets = ["2009", "2012", "2017", "ednet"]

# 为每个数据集定义多个K值候选
K_CANDIDATES = {
    # "2009": [8, 16, 24, 32],
    # "2012": [8, 16, 24, 32],
    # "2017": [8, 16, 24, 32],
    "ednet": [8, 16, 24, 32],
}

# 是否计算聚类质量指标（会增加运行时间）
COMPUTE_METRICS = True


def generate_student_cluster_multi_k(name):
    """
    生成多个K值的学生聚类文件
    - 完全符合SimQE论文公式5-6
    - 为每个K值生成独立的CSV文件
    - 可选：计算聚类质量指标辅助选择
    """
    print(f"\n{'='*70}")
    print(f"🚀 生成学生聚类 (多K值): {name}")
    print(f"{'='*70}")
    
    # ========================================================================
    # Step 1: 读取文件
    # ========================================================================
    sq_path = os.path.join(DATA_ROOT, f"{name}_final_split", "train_for_graph.csv")
    qs_path = os.path.join(DATA_ROOT, "graph_data", name, "graph", "ques_skill.csv")
    
    if not os.path.exists(sq_path) or not os.path.exists(qs_path):
        print(f"❌ 缺少文件: {sq_path} 或 {qs_path}")
        return

    df_sq = pd.read_csv(sq_path)
    df_qs = pd.read_csv(qs_path)
    
    # 标准化列名
    rename_map_sq = {}
    if "user_id" in df_sq.columns:
        rename_map_sq["user_id"] = "stu"
    if "problem_id" in df_sq.columns:
        rename_map_sq["problem_id"] = "ques"
    df_sq.rename(columns=rename_map_sq, inplace=True)
    
    print(f"   📊 学生-题目交互: {len(df_sq)} 条")
    
    # ========================================================================
    # Step 2: 关联数据
    # ========================================================================
    merged = pd.merge(df_sq, df_qs, on="ques", how="inner")
    print(f"   📊 关联技能后: {len(merged)} 条")
    
    # ========================================================================
    # Step 3: 构建学生-技能矩阵（论文公式5）
    # ========================================================================
    print("\n   ⏳ [1/3] 构建 Student-Skill 矩阵 (公式5: R_uk = 2*C - 1)...")
    
    # 计算 C_u^k (正确率)
    stu_skill_correct = merged.groupby(["stu", "skill"])["correct"].mean()
    
    # 应用公式5: R_uk = 2*C_u^k - 1
    stu_skill_R = 2 * stu_skill_correct - 1
    
    # 转换为矩阵
    stu_skill_matrix = stu_skill_R.unstack()
    
    # 填充缺失值为0（表示未做过的技能）
    stu_skill_matrix = stu_skill_matrix.fillna(0)
    
    n_students = len(stu_skill_matrix)
    n_skills = len(stu_skill_matrix.columns)
    
    print(f"       矩阵维度: ({n_students} 学生 × {n_skills} 技能)")
    print(f"       值域: [{stu_skill_matrix.min().min():.2f}, {stu_skill_matrix.max().max():.2f}]")
    
    # ========================================================================
    # Step 4: 获取训练集中的用户ID
    # ========================================================================
    # 只使用训练集中实际存在的用户ID，不进行全局补全
    all_user_ids = list(stu_skill_matrix.index)
    
    # ========================================================================
    # Step 5: 对每个K值生成聚类文件
    # ========================================================================
    k_values = K_CANDIDATES.get(name, [100])
    
    # 自动过滤不合理的K值
    valid_k_values = [k for k in k_values if k <= n_students]
    if len(valid_k_values) < len(k_values):
        removed = set(k_values) - set(valid_k_values)
        print(f"\n   ⚠️  过滤掉不合理的K值 {removed} (学生数={n_students})")
    
    print(f"\n   ⏳ [2/3] 生成多个K值的聚类文件: {valid_k_values}")
    
    # 用于存储质量指标
    metrics_summary = []
    
    for k in valid_k_values:
        print(f"\n   --- K = {k} ---")
        
        # K-Means聚类
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        clusters = kmeans.fit_predict(stu_skill_matrix)
        
        # 整理结果
        result_df = pd.DataFrame({
            "stu": stu_skill_matrix.index,
            "cluster": clusters
        })
        
        # 统计簇分布
        cluster_counts = result_df["cluster"].value_counts().sort_index()
        max_size = cluster_counts.max()
        min_size = cluster_counts.min()
        avg_size = cluster_counts.mean()
        
        print(f"       簇大小: 最大={max_size}, 最小={min_size}, 平均={avg_size:.1f}")
        
        # 计算聚类质量指标
        if COMPUTE_METRICS:
            try:
                silhouette = silhouette_score(stu_skill_matrix, clusters)
                db_index = davies_bouldin_score(stu_skill_matrix, clusters)
                
                print(f"       质量指标: Silhouette={silhouette:.3f}, DB-Index={db_index:.3f}")
                
                metrics_summary.append({
                    "K": k,
                    "silhouette": silhouette,
                    "db_index": db_index,
                    "max_size": max_size,
                    "min_size": min_size,
                    "balance_ratio": max_size / min_size
                })
            except Exception as e:
                print(f"       ⚠️ 无法计算质量指标: {e}")
        
        # 直接使用训练集中的用户ID，不进行补全
        final_df = result_df
        fill_count = 0
        
        # 保存文件（文件名包含K值）
        save_dir = os.path.join(DATA_ROOT, name, "graph")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"stu_cluster_{k}.csv")
        final_df.to_csv(save_path, index=False)
        
        print(f"       ✅ 已保存: stu_cluster_{k}.csv (总用户={len(final_df)}, 填充={fill_count})")
    
    # ========================================================================
    # Step 6: 生成质量指标汇总报告
    # ========================================================================
    if COMPUTE_METRICS and metrics_summary:
        print(f"\n   ⏳ [3/3] 生成质量指标汇总报告...")
        
        metrics_df = pd.DataFrame(metrics_summary)
        
        # 保存CSV报告
        report_path = os.path.join(DATA_ROOT, name, "graph", "stu_cluster_metrics.csv")
        metrics_df.to_csv(report_path, index=False)
        
        # 打印表格
        print(f"   === 聚类质量对比 ===")
        print(f"   {'K':<6} {'Silhouette':<12} {'DB-Index':<12} {'Balance':<10} {'建议':<10}")
        print(f"   {'-'*60}")
        
        for _, row in metrics_df.iterrows():
            k = int(row['K'])
            sil = row['silhouette']
            db = row['db_index']
            balance = row['balance_ratio']
            
            # 简单的推荐逻辑
            if sil > 0.3 and balance < 5:
                recommend = "✅ 推荐"
            elif sil > 0.2:
                recommend = "⚠️ 可用"
            else:
                recommend = "❌ 不推荐"
            
            print(f"   {k:<6} {sil:<12.3f} {db:<12.3f} {balance:<10.2f} {recommend:<10}")
        
        print(f"\n   📊 完整报告已保存: {report_path}")
        
        # 自动推荐最佳K
        best_idx = metrics_df["silhouette"].idxmax()
        best_k = int(metrics_df.loc[best_idx, "K"])
        best_sil = metrics_df.loc[best_idx, "silhouette"]
        
        print(f"\n   💡 基于Silhouette Score，推荐使用 K={best_k} (Score={best_sil:.3f})")
    
    print(f"\n{'='*70}")
    print(f"✅ {name} 完成！生成了 {len(valid_k_values)} 个聚类文件")
    print(f"{'='*70}")


def generate_all_datasets():
    """批量生成所有数据集的多K聚类文件"""
    
    print("\n" + "="*70)
    print("🎯 批量生成学生聚类 - 多K值方案")
    print("="*70)
    
    for dataset in datasets:
        # 检查 train_for_graph.csv 文件是否存在
        train_path = os.path.join(DATA_ROOT, f"{dataset}_final_split", "train_for_graph.csv")
        if os.path.exists(train_path):
            try:
                generate_student_cluster_multi_k(dataset)
            except Exception as e:
                print(f"\n❌ {dataset} 处理失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n⚠️ 跳过 {dataset}: 找不到文件 {train_path}")
    
    print("\n" + "="*70)
    print("🎉 所有数据集处理完毕！")
    print("="*70)
    
    # 生成使用指南
    print("\n📖 使用指南:")
    print("   1. 查看各数据集的 stu_cluster_metrics.csv 选择合适的K值")
    print("   2. 在 QUCUQ_Walker 中指定对应的文件:")
    print("      例如: stu_cluster_100.csv")
    print("   3. 对比不同K值对最终KT模型AUC的影响")
    print("="*70)


if __name__ == "__main__":
    generate_all_datasets()
