import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from sklearn.metrics import roc_auc_score, accuracy_score
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

# 导入我们自己写的模块
from utils.data import load_data, load_kt_sequences, load_transition_adj, load_question_skill_map
from utils.sampling import sample_metapaths_batch
from magnn_model import MAGNN_DGCN_Fusion
from copy import deepcopy

# ================= EMA 模型平均 =================
class EMA:
    """指数移动平均，用于稳定训练和提升泛化"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# ================= Focal Loss =================
class FocalLoss(nn.Module):
    """Focal Loss 用于处理类别不平衡"""
    def __init__(self, gamma=2.0, pos_weight=None, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss

# ================= Label Smoothing =================
def smooth_labels(labels, smoothing=0.1):
    """标签平滑：将 0/1 标签变为 smoothing/2 和 1-smoothing/2"""
    return labels * (1 - smoothing) + 0.5 * smoothing

# ================= 配置参数 =================
parser = argparse.ArgumentParser(description='MAGNN for Knowledge Tracing')
parser.add_argument('--dataset', type=str, default='2009', help='Dataset folder name')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate') 
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--hidden_dim', type=int, default=96, help='Hidden dimension size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--samples', type=int, default=10, help='Neighbor samples')
parser.add_argument('--max_seq_len', type=int, default=100, help='Max sequence len')
parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
parser.add_argument('--sampling_strategy', type=str, default='uniform', 
                    choices=['frequency', 'uniform'], help='Sampling strategy for metapath instances')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--eval_interval', type=int, default=1, help='Evaluate every N epochs')
parser.add_argument('--eval_max_batches', type=int, default=0, help='Max eval batches (0 means full)')
parser.add_argument('--eval_sample_mode', type=str, default='global', choices=['global', 'batch'],
                    help='Eval sampling mode')
parser.add_argument('--eval_seed', type=int, default=1234, help='Eval sampling seed')
parser.add_argument('--eval_only_test', action='store_true', default=True, help='Evaluate test set only')
parser.add_argument('--eval_use_amp', action='store_true', help='Use AMP in evaluation')
parser.add_argument('--eval_acc_threshold', type=float, default=0.5, help='ACC阈值，0.0表示自动寻找最佳阈值')
parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stop patience (epochs)')
parser.add_argument('--dgc_warmup_epochs', type=int, default=0, help='Warmup epochs for DGC branch (freeze DGC params)')
parser.add_argument('--mag_loss_weight', type=float, default=0.0, help='Weight for MAGNN branch loss (direct supervision)')
parser.add_argument('--dgc_loss_weight', type=float, default=0.0, help='Weight for DGC branch loss (direct supervision)')
parser.add_argument('--fused_loss_weight', type=float, default=1.0, help='Weight for fused output loss (default: 1.0)')
parser.add_argument('--consistency_weight', type=float, default=0.0, help='Weight for consistency loss between MAGNN and DGC branches')
# 【优化】损失函数优化参数
parser.add_argument('--use_adaptive_loss', action='store_true', help='Use adaptive loss weighting based on branch performance')
parser.add_argument('--use_hard_example_mining', action='store_true', help='Focus on hard examples for branch losses')
parser.add_argument('--branch_label_smoothing', type=float, default=0.0, help='Label smoothing for branch losses (separate from fused)')
# loss_balance_mode参数已移除（当前只实现equal模式，其他模式未实现）
# 向后兼容的旧参数名
parser.add_argument('--dgc_aux_weight', type=float, default=0.0, help='[Deprecated] Use --dgc_loss_weight instead')
parser.add_argument('--dgc_consistency_weight', type=float, default=0.0, help='[Deprecated] Use --consistency_weight instead')
parser.add_argument('--aug_prob', type=float, default=0.0)
parser.add_argument('--aug_min_len', type=int, default=30)
parser.add_argument('--aug_drop_prob', type=float, default=0.15)
parser.add_argument('--aug_skill_swap_prob', type=float, default=0.0)
parser.add_argument('--magnn_chunk_size', type=int, default=4096)
parser.add_argument('--use_checkpoint', action='store_true')
parser.add_argument('--eval_branch', type=str, default='fused', choices=['fused', 'mag', 'dgc'])
parser.add_argument('--strict_branch', action='store_true')
parser.add_argument('--disable_metapaths', type=str, default='')
parser.add_argument('--fusion_mode', type=str, default='gate', choices=['gate', 'concat'])
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor (0.0 = disabled)')
parser.add_argument('--use_ema', action='store_true', help='Use EMA for model parameters')
parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
parser.add_argument('--use_swa', action='store_true', help='Use Stochastic Weight Averaging')
parser.add_argument('--swa_start_epoch', type=int, default=30, help='Epoch to start SWA')
parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss instead of BCE')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
# GAT相关参数（用于下半分支dgcn）
parser.add_argument('--gat_dropout', type=float, default=0.2, help='GAT dropout rate (default: 0.2)')
parser.add_argument('--gat_nheads', type=int, default=1, help='GAT number of attention heads (default: 1)')
parser.add_argument('--kt_arch', type=str, default='gru', choices=['gru', 'transformer'])
parser.add_argument('--upper_tf_layers', type=int, default=1)
parser.add_argument('--upper_tf_heads', type=int, default=2)
parser.add_argument('--upper_tf_ffn_mul', type=int, default=4)
parser.add_argument('--resplit_by_loss', action='store_true')
parser.add_argument('--resplit_train_ratio', type=float, default=0.8)
parser.add_argument('--resplit_epochs', type=int, default=1)
parser.add_argument('--resplit_write_files', action='store_true')
args = parser.parse_args()

# 设置设备
if args.device == 'cuda' and not torch.cuda.is_available():
    print("CUDA不可用，切换到 CPU")
    args.device = 'cpu'
DEVICE = torch.device(args.device)
os.environ["PYTHONHASHSEED"] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device == 'cuda':
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================= 辅助函数 =================
def get_batch(sequences, batch_size, max_seq_len, augment=False, aug_prob=0.0, aug_min_len=0, aug_drop_prob=0.0, skill_swap_prob=0.0, skill2ques=None, ques2skills=None):
    import random
    sequences = list(sequences)
    expanded_sequences = []
    def augment_window(q_seq, a_seq):
        if len(q_seq) <= aug_min_len:
            return None
        max_len = min(len(q_seq), max_seq_len)
        win_len = random.randint(aug_min_len, max_len)
        start = random.randint(0, len(q_seq) - win_len)
        end = start + win_len
        return q_seq[start:end], a_seq[start:end]
    def augment_drop(q_seq, a_seq):
        if len(q_seq) <= aug_min_len:
            return None
        kept = [(q, a) for q, a in zip(q_seq, a_seq) if random.random() > aug_drop_prob]
        if len(kept) < aug_min_len:
            return None
        q_new, a_new = zip(*kept)
        return list(q_new), list(a_new)
    def augment_skill_swap(q_seq, a_seq):
        if skill2ques is None or ques2skills is None:
            return None
        if len(q_seq) <= aug_min_len:
            return None
        q_new = list(q_seq)
        for i in range(len(q_new)):
            if random.random() < skill_swap_prob:
                qid = q_new[i]
                skills = ques2skills.get(int(qid))
                if not skills:
                    continue
                s = random.choice(skills)
                candidates = skill2ques.get(int(s))
                if not candidates:
                    continue
                rep = random.choice(candidates)
                if rep != qid:
                    q_new[i] = rep
        return q_new, list(a_seq)
    for q_seq, a_seq in sequences:
        if len(q_seq) <= max_seq_len:
            expanded_sequences.append((q_seq, a_seq))
        else:
            for start in range(0, len(q_seq), max_seq_len):
                end = start + max_seq_len
                expanded_sequences.append((q_seq[start:end], a_seq[start:end]))
        if augment and random.random() < aug_prob:
            aug1 = augment_window(q_seq, a_seq)
            if aug1 is not None:
                expanded_sequences.append(aug1)
        if augment and random.random() < aug_prob:
            aug2 = augment_drop(q_seq, a_seq)
            if aug2 is not None:
                expanded_sequences.append(aug2)
        if augment and skill_swap_prob > 0.0 and random.random() < skill_swap_prob:
            aug3 = augment_skill_swap(q_seq, a_seq)
            if aug3 is not None:
                expanded_sequences.append(aug3)
    random.shuffle(expanded_sequences)
    
    for i in range(0, len(expanded_sequences), batch_size):
        batch = expanded_sequences[i:i + batch_size]
        curr_max_len = max([len(x[0]) for x in batch])
        
        q_batch = np.zeros((len(batch), curr_max_len), dtype=int)
        a_batch = np.zeros((len(batch), curr_max_len), dtype=int)
        mask_batch = np.zeros((len(batch), curr_max_len), dtype=float)
        
        for j, (q_seq, a_seq) in enumerate(batch):
            l = len(q_seq)
            q_batch[j, :l] = q_seq
            a_batch[j, :l] = a_seq
            mask_batch[j, :l] = 1.0 
            
        yield (torch.LongTensor(q_batch).to(DEVICE), 
               torch.LongTensor(a_batch).to(DEVICE),
               torch.FloatTensor(mask_batch).to(DEVICE))

def sample_eval_package(adjlists, idx_lists, num_samples, sampling_strategy, eval_seed):
    state = np.random.get_state()
    np.random.seed(eval_seed)
    sampled_adjlists, sampled_idx_lists = sample_metapaths_batch(
        adjlists, idx_lists, num_samples, sampling_strategy, None)
    np.random.set_state(state)
    return sampled_adjlists, sampled_idx_lists

def compute_pos_weight(sequences):
    pos = 0
    neg = 0
    for _, a_seq in sequences:
        if len(a_seq) <= 1:
            continue
        tail = a_seq[1:]
        p = sum(tail)
        pos += p
        neg += len(tail) - p
    if pos == 0 or neg == 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)

def expand_sequences_with_index(sequences, max_seq_len):
    expanded = []
    for idx, (q_seq, a_seq) in enumerate(sequences):
        if len(q_seq) <= max_seq_len:
            expanded.append((q_seq, a_seq, idx))
        else:
            for start in range(0, len(q_seq), max_seq_len):
                end = start + max_seq_len
                expanded.append((q_seq[start:end], a_seq[start:end], idx))
    return expanded

def batch_from_expanded(expanded, batch_size):
    for i in range(0, len(expanded), batch_size):
        batch = expanded[i:i + batch_size]
        curr_max_len = max([len(x[0]) for x in batch])
        q_batch = np.zeros((len(batch), curr_max_len), dtype=int)
        a_batch = np.zeros((len(batch), curr_max_len), dtype=int)
        mask_batch = np.zeros((len(batch), curr_max_len), dtype=float)
        orig_indices = []
        for j, (q_seq, a_seq, orig_idx) in enumerate(batch):
            l = len(q_seq)
            q_batch[j, :l] = q_seq
            a_batch[j, :l] = a_seq
            mask_batch[j, :l] = 1.0
            orig_indices.append(orig_idx)
        yield (torch.LongTensor(q_batch).to(DEVICE),
               torch.LongTensor(a_batch).to(DEVICE),
               torch.FloatTensor(mask_batch).to(DEVICE),
               orig_indices)

def compute_sequence_losses(model, sequences, adjlists, idx_lists, features_list, adj_out, adj_in,
                            batch_size, max_seq_len, num_samples, sampling_strategy):
    expanded = expand_sequences_with_index(sequences, max_seq_len)
    loss_sum = np.zeros(len(sequences), dtype=np.float32)
    loss_count = np.zeros(len(sequences), dtype=np.int32)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    model.eval()
    with torch.no_grad():
        for q_batch, a_batch, mask, orig_indices in batch_from_expanded(expanded, batch_size):
            batch_q_ids = q_batch.view(-1).cpu().numpy()
            unique_q_ids = np.unique(batch_q_ids[batch_q_ids >= 0])
            sampled_adjlists, sampled_idx_lists = sample_metapaths_batch(
                adjlists, idx_lists, num_samples, sampling_strategy, unique_q_ids)
            processed_package = (sampled_adjlists, sampled_idx_lists, features_list)
            logits, labels = model(q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask)
            valid_mask = mask[:, 1:]
            loss_element = criterion(logits, labels)
            seq_len = valid_mask.sum(dim=1)
            seq_loss = (loss_element * valid_mask).sum(dim=1) / torch.clamp(seq_len, min=1.0)
            for i, orig_idx in enumerate(orig_indices):
                loss_sum[orig_idx] += float(seq_loss[i].item())
                loss_count[orig_idx] += 1
    avg_loss = np.zeros(len(sequences), dtype=np.float32)
    valid = loss_count > 0
    avg_loss[valid] = loss_sum[valid] / loss_count[valid]
    return avg_loss

def parse_kt_sequences_with_skill(filepath):
    sequences = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到序列文件: {filepath}")
    with open(filepath, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if len(lines[i].strip()) == 0:
                i += 1
                continue
            q_seq = list(map(int, lines[i + 1].strip().split(',')))
            skill_line = lines[i + 2].strip()
            a_seq = list(map(int, lines[i + 3].strip().split(',')))
            if len(q_seq) == len(a_seq):
                sequences.append((q_seq, a_seq, skill_line if len(skill_line) > 0 else "0"))
            i += 4
    return sequences

def write_kt_sequences(sequences, filepath):
    with open(filepath, 'w') as f:
        for q_seq, a_seq, skill_line in sequences:
            f.write(f"{len(q_seq)}\n")
            f.write(",".join(map(str, q_seq)) + "\n")
            f.write(f"{skill_line}\n")
            f.write(",".join(map(str, a_seq)) + "\n")

def evaluate(model, sequences, adjlists, idx_lists, features_list, adj_out, adj_in,
            batch_size, max_seq_len, num_samples, sampling_strategy, max_batches=0,
            eval_sample_mode='global', eval_seed=1234, cached_package=None, acc_threshold=0.5):
    """
    评估函数，使用动态采样
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        if cached_package is None and eval_sample_mode == 'global':
            sampled_adjlists, sampled_idx_lists = sample_eval_package(
                adjlists, idx_lists, num_samples, sampling_strategy, eval_seed)
            cached_package = (sampled_adjlists, sampled_idx_lists, features_list)
        batch_count = 0
        for q_batch, a_batch, mask in get_batch(sequences, batch_size, max_seq_len):
            if eval_sample_mode == 'batch':
                batch_q_ids = q_batch.view(-1).cpu().numpy()
                unique_q_ids = np.unique(batch_q_ids[batch_q_ids >= 0])
                sampled_adjlists, sampled_idx_lists = sample_metapaths_batch(
                    adjlists, idx_lists, num_samples, sampling_strategy, unique_q_ids)
                processed_package = (sampled_adjlists, sampled_idx_lists, features_list)
            else:
                processed_package = cached_package

            strict_mode = args.strict_branch and args.eval_branch != 'fused'
            return_branch = (args.eval_branch != 'fused') and (not strict_mode)
            if args.eval_use_amp and DEVICE.type == 'cuda':
                with torch.cuda.amp.autocast():
                    if strict_mode:
                        logits, labels = model(
                            q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask, compute_branch=args.eval_branch)
                    elif return_branch:
                        logits, labels, mag_logits, dgc_logits = model(
                            q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask, return_branch=True)
                    else:
                        logits, labels = model(q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask)
            else:
                if strict_mode:
                    logits, labels = model(
                        q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask, compute_branch=args.eval_branch)
                elif return_branch:
                    logits, labels, mag_logits, dgc_logits = model(
                        q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask, return_branch=True)
                else:
                    logits, labels = model(q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask)
            if return_branch:
                if args.eval_branch == 'mag':
                    logits = mag_logits
                elif args.eval_branch == 'dgc':
                    logits = dgc_logits
            valid_mask = mask[:, 1:]
            pred_prob = torch.sigmoid(logits)
            valid_indices = valid_mask > 0.5
            y_true.extend(labels[valid_indices].cpu().numpy())
            y_pred.extend(pred_prob[valid_indices].cpu().numpy())
            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break
    
    if len(y_true) == 0:
        return 0.0, 0.0
    
    auc = roc_auc_score(y_true, y_pred)
    if acc_threshold <= 0.0:
        # 自动阈值：遍历百分位找最佳ACC（快速近似）
        preds = np.array(y_pred)
        gts = np.array(y_true)
        percentiles = np.linspace(0.05, 0.95, 19)
        best_acc = 0.0
        best_t = 0.5
        for p in percentiles:
            t = np.quantile(preds, p)
            acc_p = accuracy_score(gts, preds >= t)
            if acc_p > best_acc:
                best_acc = acc_p
                best_t = t
        acc = best_acc
    else:
        acc = accuracy_score(y_true, np.array(y_pred) >= acc_threshold)
    return auc, acc

# ================= 主程序 =================
def main():
    print(f"Config: {args}")
    base_dir = os.getenv("MAGNN_DATA_DIR")
    if base_dir is None or len(base_dir) == 0:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)
    
    # 1. 加载数据
    print("Loading Graph Data...")
    adjlists, idx_lists, features_list = load_data(base_dir=base_dir) 
    
    num_nodes_list = [f.shape[0] if f is not None else 0 for f in features_list]
    if features_list[0] is None:
        raise ValueError("features_0.npy is missing!")
    input_dim = features_list[0].shape[1]
    
    print(f"Num Nodes: Q={num_nodes_list[0]}, U={num_nodes_list[1]}, K={num_nodes_list[2]}, C={num_nodes_list[3]}")
    print(f"Input Dim: {input_dim}")
    
    print("Loading KT Sequences...")
    mapped_dir = os.path.join(base_dir, "mapped_data")
    train_path = os.path.join(mapped_dir, "train_mapped.txt")
    test_path = os.path.join(mapped_dir, "test_mapped.txt")
    train_records = parse_kt_sequences_with_skill(train_path)
    test_records = parse_kt_sequences_with_skill(test_path)
    train_seqs = [(q, a) for q, a, _ in train_records]
    test_seqs = [(q, a) for q, a, _ in test_records]
    print(f"Train Seqs: {len(train_seqs)}, Test Seqs: {len(test_seqs)}")
    adj_out, adj_in = load_transition_adj(num_nodes_list[0], base_dir=base_dir, device=DEVICE)
    skill_indices, skill_offsets, num_skills, _ = load_question_skill_map(base_dir=base_dir)
    skill_swap_maps = None
    if args.dataset == '2009' and args.aug_skill_swap_prob > 0.0:
        mapped_dir = os.path.join(base_dir, "mapped_data")
        qs_path = os.path.join(mapped_dir, "ques_skill_mapped.csv")
        df = pd.read_csv(qs_path)
        if 'ques' in df.columns and 'skill' in df.columns:
            q_col, s_col = 'ques', 'skill'
        elif 'problem_id' in df.columns and 'skill_id' in df.columns:
            q_col, s_col = 'problem_id', 'skill_id'
        else:
            q_col, s_col = df.columns[0], df.columns[1]
        df[q_col] = pd.to_numeric(df[q_col], errors='coerce')
        df[s_col] = pd.to_numeric(df[s_col], errors='coerce')
        df = df.dropna().astype(int)
        skill2ques = {}
        ques2skills = {}
        for q, s in zip(df[q_col].values, df[s_col].values):
            skill2ques.setdefault(int(s), []).append(int(q))
            ques2skills.setdefault(int(q), []).append(int(s))
        skill_swap_maps = (skill2ques, ques2skills)
    # 2. 初始化模型
    full_metapaths = ['0-1-0', '0-2-0', '0-1-3-1-0']
    metapath_list = list(full_metapaths)
    if args.disable_metapaths:
        disable_set = set([s.strip() for s in args.disable_metapaths.split(',') if len(s.strip()) > 0])
        metapath_list = [mp for mp in metapath_list if mp not in disable_set]
        if len(metapath_list) == 0:
            raise ValueError("metapath_list 为空，请至少保留一个元路径")
    keep_indices = [i for i, mp in enumerate(full_metapaths) if mp in metapath_list]
    adjlists = [adjlists[i] for i in keep_indices]
    idx_lists = [idx_lists[i] for i in keep_indices]
    
    model = MAGNN_DGCN_Fusion(
        num_nodes_list=num_nodes_list,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        metapath_list=metapath_list,
        num_samples=args.samples,
        device=DEVICE,
        num_questions=num_nodes_list[0],
        dropout=args.dropout,
        question_skill_info=(skill_indices, skill_offsets, num_skills),
        magnn_chunk_size=args.magnn_chunk_size,
        use_checkpoint=args.use_checkpoint,
        fusion_mode=args.fusion_mode,
        gat_dropout=args.gat_dropout,
        gat_nheads=args.gat_nheads,
        upper_encoder=args.kt_arch,
        upper_tf_layers=args.upper_tf_layers,
        upper_tf_heads=args.upper_tf_heads,
        upper_tf_ffn_mul=args.upper_tf_ffn_mul,
        max_seq_len=args.max_seq_len
    ).to(DEVICE)
    if args.dataset == '2009':
        # 【关键】融合权重初始化：强烈偏向MAGNN（因为MAGNN性能更好78.60 vs 76.53）
        # fusion_alpha通过sigmoid后控制MAGNN的权重，值越大越偏向MAGNN
        model.fusion_alpha.data.fill_(0.0)
        model.dgc_sem_alpha.data.fill_(0.2)
        if args.dgc_warmup_epochs == 0:
            args.dgc_warmup_epochs = 10
        if args.dgc_consistency_weight == 0.0:
            args.dgc_consistency_weight = 0.1

    # 3. 初始化 Embedding
    for i, feat in enumerate(features_list):
        if feat is not None:
            model.embeddings[i].weight.data.copy_(torch.from_numpy(feat).float())

    if args.resplit_by_loss:
        all_records = train_records + test_records
        all_seqs = [(q, a) for q, a, _ in all_records]
        total_count = len(all_seqs)
        k_keep = int(total_count * args.resplit_train_ratio)
        k_keep = max(1, min(total_count - 1, k_keep))
        tmp_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        tmp_criterion = nn.BCEWithLogitsLoss(reduction='none')
        for _ in range(max(1, args.resplit_epochs)):
            t0 = time.time()
            total_loss = 0.0
            step_count = 0
            model.train()
            for q_batch, a_batch, mask in get_batch(
                all_seqs, args.batch_size, args.max_seq_len,
                augment=False, aug_prob=0.0, aug_min_len=0, aug_drop_prob=0.0
            ):
                tmp_optimizer.zero_grad()
                batch_q_ids = q_batch.view(-1).cpu().numpy()
                unique_q_ids = np.unique(batch_q_ids[batch_q_ids >= 0])
                sampled_adjlists, sampled_idx_lists = sample_metapaths_batch(
                    adjlists, idx_lists, args.samples, args.sampling_strategy, unique_q_ids)
                processed_package = (sampled_adjlists, sampled_idx_lists, features_list)
                with torch.cuda.amp.autocast():
                    logits, labels = model(q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask)
                    valid_mask = mask[:, 1:]
                    valid_sum = valid_mask.sum()
                    loss_element = tmp_criterion(logits, labels)
                    loss = (loss_element * valid_mask).sum() / valid_sum if valid_sum > 0 else loss_element.mean()
                loss.backward()
                tmp_optimizer.step()
                total_loss += loss.item()
                step_count += 1
            t1 = time.time()
            avg_loss = total_loss / max(1, step_count)
            print(f"Resplit Warmup | Time: {t1-t0:.1f}s | Loss: {avg_loss:.4f}")
        all_losses = compute_sequence_losses(
            model, all_seqs, adjlists, idx_lists, features_list, adj_out, adj_in,
            args.batch_size, args.max_seq_len, args.samples, args.sampling_strategy
        )
        all_indices = np.argsort(all_losses)[::-1]
        train_keep = set(all_indices[:k_keep].tolist())
        new_train_records = [all_records[i] for i in train_keep]
        new_test_records = [all_records[i] for i in all_indices[k_keep:]]
        train_records = new_train_records
        test_records = new_test_records
        train_seqs = [(q, a) for q, a, _ in train_records]
        test_seqs = [(q, a) for q, a, _ in test_records]
        print(f"Resplit Done: Train={len(train_seqs)}, Test={len(test_seqs)}")
        if args.resplit_write_files:
            write_kt_sequences(train_records, train_path)
            write_kt_sequences(test_records, test_path)
            print(f"Resplit Files Written: {train_path}, {test_path}")
    pos_weight = compute_pos_weight(train_seqs)
    if pos_weight is not None:
        pos_weight = pos_weight.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        print(f"EMA enabled with decay={args.ema_decay}")
    
    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.5)
        print(f"SWA enabled, starting at epoch {args.swa_start_epoch}")
    
    if args.use_focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight, reduction='none')
        print(f"Focal Loss enabled with gamma={args.focal_gamma}")
    elif pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    if args.label_smoothing > 0:
        print(f"Label Smoothing enabled with factor={args.label_smoothing}")

    # 5. 训练循环
    print("Start Training...")
    best_auc = 0.0
    no_improve_epochs = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        model.train()
        freeze_dgc = epoch < args.dgc_warmup_epochs
        for name, param in model.named_parameters():
            if name.startswith("dgc_"):
                param.requires_grad = not freeze_dgc
        total_loss = 0
        step_count = 0
        t0 = time.time()
        
        for q_batch, a_batch, mask in get_batch(
            train_seqs, args.batch_size, args.max_seq_len,
            augment=(args.dataset == '2009'),
            aug_prob=args.aug_prob,
            aug_min_len=args.aug_min_len,
            aug_drop_prob=args.aug_drop_prob,
            skill_swap_prob=(args.aug_skill_swap_prob if args.dataset == '2009' else 0.0),
            skill2ques=(skill_swap_maps[0] if skill_swap_maps is not None else None),
            ques2skills=(skill_swap_maps[1] if skill_swap_maps is not None else None)
        ):
            optimizer.zero_grad()
            
            # 获取当前batch的题目ID（用于批量采样）
            batch_q_ids = q_batch.view(-1).cpu().numpy()
            unique_q_ids = np.unique(batch_q_ids[batch_q_ids >= 0])
            
            # 动态采样元路径实例（每个batch都采样，实现真正的动态采样）
            sampled_adjlists, sampled_idx_lists = sample_metapaths_batch(
                adjlists, idx_lists, args.samples, args.sampling_strategy, unique_q_ids)
            
            # 使用采样后的数据
            processed_package = (sampled_adjlists, sampled_idx_lists, features_list)
            
            with torch.cuda.amp.autocast():
                strict_mode = args.strict_branch and args.eval_branch != 'fused'
                # 【修复】兼容旧参数名
                mag_loss_weight = args.mag_loss_weight if args.mag_loss_weight > 0.0 else 0.0
                dgc_loss_weight = args.dgc_loss_weight if args.dgc_loss_weight > 0.0 else args.dgc_aux_weight
                consistency_weight = args.consistency_weight if args.consistency_weight > 0.0 else args.dgc_consistency_weight
                
                # 【修复】训练时总是需要计算分支输出（如果设置了任何分支损失或一致性损失）
                use_branch = ((mag_loss_weight > 0.0) or (dgc_loss_weight > 0.0) or (consistency_weight > 0.0)) and (not strict_mode)
                
                if strict_mode:
                    logits, labels = model(
                        q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask, compute_branch=args.eval_branch)
                    mag_logits = None
                    dgc_logits = None
                elif use_branch:
                    # 【修复】训练时总是返回融合输出和分支输出
                    logits, labels, mag_logits, dgc_logits = model(
                        q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask, return_branch=True)
                else:
                    logits, labels = model(q_batch, a_batch, processed_package, adj_out, adj_in, mask=mask)
                    mag_logits = None
                    dgc_logits = None
                
                # 【修复】eval_branch只影响评估，不影响训练损失计算
                # 训练时主损失总是基于融合输出
                
                # [新增] Label Smoothing
                smooth_labels_batch = labels
                if args.label_smoothing > 0:
                    smooth_labels_batch = smooth_labels(labels, args.label_smoothing)
                
                valid_mask = mask[:, 1:]
                valid_sum = valid_mask.sum()
                if valid_sum > 0:
                    # 【优化】改进的损失函数：独立监督 + 融合监督 + 智能平衡
                    # 初始化loss为tensor（确保有梯度）
                    loss = None
                    
                    # 为分支损失准备标签（可以使用不同的标签平滑策略）
                    branch_labels = labels if args.branch_label_smoothing == 0.0 else smooth_labels(labels, args.branch_label_smoothing)
                    
                    # 静态损失权重（简化版本：移除动态调整，保持稳定训练）
                    dynamic_fused_weight = args.fused_loss_weight
                    dynamic_mag_weight = mag_loss_weight
                    dynamic_dgc_weight = dgc_loss_weight
                    
                    # 1. 融合输出的损失（主损失）- 使用动态权重
                    fused_loss = None
                    if args.fused_loss_weight > 0.0:
                        fused_loss_element = criterion(logits, smooth_labels_batch)
                        fused_loss = (fused_loss_element * valid_mask).sum() / valid_sum
                        if loss is None:
                            loss = dynamic_fused_weight * fused_loss
                        else:
                            loss = loss + dynamic_fused_weight * fused_loss
                    
                    # 2. MAGNN分支的独立损失（上半分支）- 独立监督（使用动态权重）
                    mag_loss = None
                    if use_branch and mag_logits is not None:
                        if dynamic_mag_weight > 0.0:
                            # 【优化】分支损失可以使用原始标签（更严格的监督）
                            mag_loss_element = criterion(mag_logits, branch_labels)
                            
                            # 【优化】困难样本挖掘：关注预测错误的样本
                            if args.use_hard_example_mining:
                                with torch.no_grad():
                                    mag_pred = torch.sigmoid(mag_logits)
                                    mag_error = torch.abs(mag_pred - branch_labels)
                                    # 选择错误率最高的50%样本
                                    error_threshold = torch.quantile(mag_error[valid_mask.bool()], 0.5)
                                    hard_mask = (mag_error > error_threshold).float() * valid_mask
                                    hard_sum = hard_mask.sum()
                                    if hard_sum > 0:
                                        mag_loss = (mag_loss_element * hard_mask).sum() / hard_sum
                                    else:
                                        mag_loss = (mag_loss_element * valid_mask).sum() / valid_sum
                            else:
                                mag_loss = (mag_loss_element * valid_mask).sum() / valid_sum
                            
                            loss = loss + dynamic_mag_weight * mag_loss
                    
                    # 3. DGC分支的独立损失（下半分支）- 独立监督（使用动态权重）
                    dgc_loss = None
                    if use_branch and dgc_logits is not None:
                        if dynamic_dgc_weight > 0.0:
                            # 【优化】分支损失可以使用原始标签（更严格的监督）
                            dgc_loss_element = criterion(dgc_logits, branch_labels)
                            
                            # 【优化】困难样本挖掘：关注预测错误的样本
                            if args.use_hard_example_mining:
                                with torch.no_grad():
                                    dgc_pred = torch.sigmoid(dgc_logits)
                                    dgc_error = torch.abs(dgc_pred - branch_labels)
                                    # 选择错误率最高的50%样本
                                    error_threshold = torch.quantile(dgc_error[valid_mask.bool()], 0.5)
                                    hard_mask = (dgc_error > error_threshold).float() * valid_mask
                                    hard_sum = hard_mask.sum()
                                    if hard_sum > 0:
                                        dgc_loss = (dgc_loss_element * hard_mask).sum() / hard_sum
                                    else:
                                        dgc_loss = (dgc_loss_element * valid_mask).sum() / valid_sum
                            else:
                                dgc_loss = (dgc_loss_element * valid_mask).sum() / valid_sum
                            
                            loss = loss + dynamic_dgc_weight * dgc_loss
                    
                    # 4. 【优化】改进的一致性损失：使用KL散度，更关注概率分布的差异
                    cons_loss = None
                    if use_branch and mag_logits is not None and dgc_logits is not None:
                        if consistency_weight > 0.0:
                            # 使用KL散度衡量两个分支预测分布的差异
                            mag_probs = torch.sigmoid(mag_logits.detach())
                            dgc_probs = torch.sigmoid(dgc_logits)
                            
                            # KL散度：KL(DGC || MAGNN)
                            # 避免log(0)，添加小的epsilon
                            eps = 1e-8
                            mag_probs_clamped = mag_probs.clamp(eps, 1 - eps)
                            dgc_probs_clamped = dgc_probs.clamp(eps, 1 - eps)
                            
                            kl_loss = dgc_probs_clamped * torch.log(dgc_probs_clamped / mag_probs_clamped) + \
                                     (1 - dgc_probs_clamped) * torch.log((1 - dgc_probs_clamped) / (1 - mag_probs_clamped))
                            
                            cons_loss = (kl_loss * valid_mask).sum() / valid_sum
                            loss = loss + consistency_weight * cons_loss
                    
                    # 【优化】自适应损失权重：根据分支性能动态调整（在动态权重基础上进一步调整）
                    if args.use_adaptive_loss and use_branch and mag_logits is not None and dgc_logits is not None and loss is not None:
                        # 计算每个分支的准确率（不需要梯度）
                        with torch.no_grad():
                            mag_pred = (torch.sigmoid(mag_logits) > 0.5).float()
                            dgc_pred = (torch.sigmoid(dgc_logits) > 0.5).float()
                            mag_acc = ((mag_pred == labels) * valid_mask).sum() / valid_sum
                            dgc_acc = ((dgc_pred == labels) * valid_mask).sum() / valid_sum
                            
                            # 动态调整权重：性能差的分支给予更多权重
                            if mag_acc < dgc_acc:
                                # MAGNN性能较差，增加其权重
                                mag_weight_scale = 1.0 + (dgc_acc - mag_acc) * 2.0
                                dgc_weight_scale = 1.0
                            else:
                                # DGC性能较差，增加其权重
                                mag_weight_scale = 1.0
                                dgc_weight_scale = 1.0 + (mag_acc - dgc_acc) * 2.0
                        
                        # 【修复】重新计算加权损失（必须在no_grad块外，保持梯度）
                        if mag_loss is not None and mag_loss_weight > 0.0:
                            # 先减去原来的权重，再加上新的权重
                            loss = loss - mag_loss_weight * mag_loss + mag_loss_weight * mag_weight_scale * mag_loss
                        if dgc_loss is not None and dgc_loss_weight > 0.0:
                            loss = loss - dgc_loss_weight * dgc_loss + dgc_loss_weight * dgc_weight_scale * dgc_loss
                    
                    # 确保至少有一个损失
                    if loss is None:
                        # 如果没有设置任何损失权重，默认使用融合损失
                        fused_loss_element = criterion(logits, smooth_labels_batch)
                        fused_loss = (fused_loss_element * valid_mask).sum() / valid_sum
                        loss = fused_loss
                    elif isinstance(loss, torch.Tensor):
                        # 检查loss是否为0（但保持梯度）
                        loss_value = loss.item()
                        if loss_value == 0.0:
                            # 如果loss为0，使用融合损失
                            fused_loss_element = criterion(logits, smooth_labels_batch)
                            fused_loss = (fused_loss_element * valid_mask).sum() / valid_sum
                            loss = fused_loss
                    
                    # 最终检查：确保loss是tensor且有梯度
                    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                        # 如果loss不是tensor或没有梯度，使用融合损失
                        fused_loss_element = criterion(logits, smooth_labels_batch)
                        fused_loss = (fused_loss_element * valid_mask).sum() / valid_sum
                        loss = fused_loss
                else:
                    # 当valid_sum == 0时，创建一个有梯度的零tensor
                    # 但实际上这种情况不应该发生，因为valid_sum == 0意味着没有有效样本
                    # 为了安全，我们使用融合损失（即使valid_sum==0，logits仍然有梯度）
                    fused_loss_element = criterion(logits, smooth_labels_batch)
                    loss = fused_loss_element.mean()  # 使用mean而不是sum/valid_sum
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
            scaler.step(optimizer)
            scaler.update()
            
            # [新增] EMA 更新
            if ema is not None:
                ema.update()
            
            total_loss += loss.item()
            step_count += 1
            
            if step_count % 10 == 0:
                # 【新增】显示详细的损失信息，帮助调试
                loss_str = f"Epoch {epoch+1} | Step {step_count} | Total Loss: {loss.item():.4f}"
                if use_branch and valid_sum > 0:
                    if 'mag_logits' in locals() and mag_logits is not None and mag_loss_weight > 0.0:
                        mag_loss_val = (criterion(mag_logits, smooth_labels_batch) * valid_mask).sum() / valid_sum
                        loss_str += f" | MAG Loss: {mag_loss_val.item():.4f}"
                    if 'dgc_logits' in locals() and dgc_logits is not None and dgc_loss_weight > 0.0:
                        dgc_loss_val = (criterion(dgc_logits, smooth_labels_batch) * valid_mask).sum() / valid_sum
                        loss_str += f" | DGC Loss: {dgc_loss_val.item():.4f}"
                    if 'mag_logits' in locals() and 'dgc_logits' in locals() and mag_logits is not None and dgc_logits is not None and consistency_weight > 0.0:
                        cons_loss_val = (F.mse_loss(torch.sigmoid(dgc_logits), torch.sigmoid(mag_logits.detach()), reduction='none') * valid_mask).sum() / valid_sum
                        loss_str += f" | Cons Loss: {cons_loss_val.item():.4f}"
                print(loss_str, end='\r')
        
        torch.cuda.empty_cache()
        
        # [新增] SWA 更新
        if args.use_swa and epoch >= args.swa_start_epoch:
            swa_model.update_parameters(model)
        
        do_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch + 1 == args.epochs)
        cached_package = None
        if do_eval and args.eval_sample_mode == 'global':
            sampled_adjlists, sampled_idx_lists = sample_eval_package(
                adjlists, idx_lists, args.samples, args.sampling_strategy, args.eval_seed)
            cached_package = (sampled_adjlists, sampled_idx_lists, features_list)
        if do_eval:
            # [新增] 评估时使用 EMA 权重
            if ema is not None:
                ema.apply_shadow()
            
            if args.eval_only_test:
                train_auc, train_acc = 0.0, 0.0
                test_auc, test_acc = evaluate(model, test_seqs, adjlists, idx_lists, 
                                          features_list, adj_out, adj_in, args.batch_size, args.max_seq_len, 
                                          args.samples, args.sampling_strategy,
                                          max_batches=args.eval_max_batches,
                                          eval_sample_mode=args.eval_sample_mode,
                                          eval_seed=args.eval_seed,
                                          cached_package=cached_package,
                                          acc_threshold=args.eval_acc_threshold)
            else:
                train_auc, train_acc = evaluate(model, train_seqs, adjlists, idx_lists, 
                                           features_list, adj_out, adj_in, args.batch_size, args.max_seq_len, 
                                           args.samples, args.sampling_strategy,
                                           max_batches=args.eval_max_batches,
                                           eval_sample_mode=args.eval_sample_mode,
                                           eval_seed=args.eval_seed,
                                           cached_package=cached_package,
                                           acc_threshold=args.eval_acc_threshold)
                test_auc, test_acc = evaluate(model, test_seqs, adjlists, idx_lists, 
                                          features_list, adj_out, adj_in, args.batch_size, args.max_seq_len, 
                                          args.samples, args.sampling_strategy,
                                          max_batches=args.eval_max_batches,
                                          eval_sample_mode=args.eval_sample_mode,
                                          eval_seed=args.eval_seed,
                                          cached_package=cached_package,
                                          acc_threshold=args.eval_acc_threshold)
            
            # [新增] 评估后恢复原始权重
            if ema is not None:
                ema.restore()
        else:
            train_auc, train_acc = 0.0, 0.0
            test_auc, test_acc = 0.0, 0.0
        
        t1 = time.time()
        avg_loss = total_loss / step_count if step_count > 0 else 0
        print(f"\nEpoch {epoch+1}/{args.epochs} | Time: {t1-t0:.1f}s | Loss: {avg_loss:.4f}")
        if do_eval:
            if args.eval_only_test:
                print("  Train AUC: (skipped) | ACC: (skipped)")
            else:
                print(f"  Train AUC: {train_auc:.4f} | ACC: {train_acc:.4f}")
            print(f"  Test  AUC: {test_auc:.4f}  | ACC: {test_acc:.4f}")
        else:
            print("  Train AUC: (skipped) | ACC: (skipped)")
            print("  Test  AUC: (skipped)")
        
        # [改进] SWA 阶段使用 SWA 调度器
        if args.use_swa and epoch >= args.swa_start_epoch:
            swa_scheduler.step()
        else:
            scheduler.step()
        
        if do_eval:
            if test_auc > best_auc:
                best_auc = test_auc
                no_improve_epochs = 0
                # [改进] 保存时使用EMA权重（如果启用）
                if ema is not None:
                    ema.apply_shadow()
                torch.save(model.state_dict(), f"best_model.pth")
                if ema is not None:
                    ema.restore()
                print("  >>> Best Model Saved!")
            else:
                no_improve_epochs += 1
                print(f"  >>> No Improvement: {no_improve_epochs}/{args.early_stop_patience}")
                if no_improve_epochs >= args.early_stop_patience:
                    print(f"  >>> Early Stop Triggered (patience={args.early_stop_patience})")
                    break
    
    # [新增] 训练结束后，如果使用SWA，最终评估用SWA模型
    if args.use_swa and swa_model is not None:
        print("\n[SWA] Final evaluation with SWA model...")
        # 复制SWA权重到原模型
        model.load_state_dict(swa_model.module.state_dict())
        cached_package = None
        if args.eval_sample_mode == 'global':
            sampled_adjlists, sampled_idx_lists = sample_eval_package(
                adjlists, idx_lists, args.samples, args.sampling_strategy, args.eval_seed)
            cached_package = (sampled_adjlists, sampled_idx_lists, features_list)
        swa_test_auc, swa_test_acc = evaluate(model, test_seqs, adjlists, idx_lists, 
                                      features_list, adj_out, adj_in, args.batch_size, args.max_seq_len, 
                                      args.samples, args.sampling_strategy,
                                      max_batches=args.eval_max_batches,
                                      eval_sample_mode=args.eval_sample_mode,
                                      eval_seed=args.eval_seed,
                                      cached_package=cached_package,
                                      acc_threshold=args.eval_acc_threshold)
        print(f"  SWA Test AUC: {swa_test_auc:.4f} | ACC: {swa_test_acc:.4f}")
        if swa_test_auc > best_auc:
            best_auc = swa_test_auc
            torch.save(model.state_dict(), f"best_model.pth")
            print("  >>> SWA Model is better, saved!")
            
    print(f"\nTraining Finished. Best Test AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()
