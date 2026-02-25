import numpy as np
import random


def sample_metapaths(adjlists, idx_lists, num_samples, sampling_strategy='frequency'):
    """
    从实例池中动态采样元路径实例
    
    Args:
        adjlists: 邻接表列表，每个元素是一个列表，包含每个节点的邻居
        idx_lists: 元路径实例索引列表，每个元素是一个列表，包含每个节点的元路径实例
        num_samples: 每个节点采样的邻居数量
        sampling_strategy: 采样策略 ('frequency' 或 'uniform')
    
    Returns:
        sampled_adjlists: 采样后的邻接表
        sampled_idx_lists: 采样后的元路径实例索引
    """
    sampled_adjlists = []
    sampled_idx_lists = []
    
    for adjlist, idx_list in zip(adjlists, idx_lists):
        sampled_adj = []
        sampled_idx = []
        
        for neighbors, metapaths in zip(adjlist, idx_list):
            # 如果没有邻居，直接跳过
            if len(neighbors) == 0:
                sampled_adj.append(neighbors)
                sampled_idx.append(metapaths)
                continue
            
            # 根据采样策略选择采样方法
            if sampling_strategy == 'frequency':
                # 基于频率的下采样
                unique, counts = np.unique(neighbors, return_counts=True)
                p = []
                for count in counts:
                    # 频率越高，采样概率越低（下采样）
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                
                # 随机采样指定数量的邻居
                samples = min(num_samples, len(neighbors))
                sampled_idx_pos = np.sort(np.random.choice(
                    len(neighbors), samples, replace=False, p=p))
            else:
                # 均匀随机采样
                samples = min(num_samples, len(neighbors))
                sampled_idx_pos = np.sort(np.random.choice(
                    len(neighbors), samples, replace=False))
            
            # 构建采样的结果
            sampled_neighbors = [neighbors[i] for i in sampled_idx_pos]
            sampled_metapaths = metapaths[sampled_idx_pos] if len(metapaths) > 0 else metapaths
            
            sampled_adj.append(sampled_neighbors)
            sampled_idx.append(sampled_metapaths)
        
        sampled_adjlists.append(sampled_adj)
        sampled_idx_lists.append(sampled_idx)
    
    return sampled_adjlists, sampled_idx_lists


def sample_metapaths_batch(adjlists, idx_lists, num_samples, sampling_strategy='frequency', 
                          batch_q_ids=None):
    """
    批量采样元路径实例（针对特定batch的题目）
    
    Args:
        adjlists: 邻接表列表
        idx_lists: 元路径实例索引列表
        num_samples: 每个节点采样的邻居数量
        sampling_strategy: 采样策略
        batch_q_ids: 当前batch的题目ID列表（可选，用于只采样相关节点的元路径）
    
    Returns:
        sampled_adjlists: 采样后的邻接表
        sampled_idx_lists: 采样后的元路径实例索引
    """
    if batch_q_ids is None:
        # 如果没有指定batch_q_ids，使用全局采样
        return sample_metapaths(adjlists, idx_lists, num_samples, sampling_strategy)
    
    batch_q_ids_set = set(batch_q_ids)
    
    sampled_adjlists = []
    sampled_idx_lists = []
    
    for adjlist, idx_list in zip(adjlists, idx_lists):
        sampled_adj = []
        sampled_idx = []
        
        for node_idx, (neighbors, metapaths) in enumerate(zip(adjlist, idx_list)):
            # 如果节点不在当前batch中，直接使用原始数据（不采样）
            if node_idx not in batch_q_ids_set:
                sampled_adj.append(neighbors)
                sampled_idx.append(metapaths)
                continue
            
            # 如果节点在当前batch中，进行采样
            if len(neighbors) == 0:
                sampled_adj.append(neighbors)
                sampled_idx.append(metapaths)
                continue
            
            # 根据采样策略选择采样方法
            if sampling_strategy == 'frequency':
                unique, counts = np.unique(neighbors, return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                
                samples = min(num_samples, len(neighbors))
                sampled_idx_pos = np.sort(np.random.choice(
                    len(neighbors), samples, replace=False, p=p))
            else:
                samples = min(num_samples, len(neighbors))
                sampled_idx_pos = np.sort(np.random.choice(
                    len(neighbors), samples, replace=False))
            
            sampled_neighbors = [neighbors[i] for i in sampled_idx_pos]
            sampled_metapaths = metapaths[sampled_idx_pos] if len(metapaths) > 0 else metapaths
            
            sampled_adj.append(sampled_neighbors)
            sampled_idx.append(sampled_metapaths)
        
        sampled_adjlists.append(sampled_adj)
        sampled_idx_lists.append(sampled_idx)
    
    return sampled_adjlists, sampled_idx_lists
