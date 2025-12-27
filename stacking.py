import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
import re
import os
import time
import gc
import threading
from queue import Queue


# ==========================================
# 1. 基础组件 (Dataset & Model)
# ==========================================
class SparseRTDataset(Dataset):
    def __init__(self, sparse_matrix, labels=None):
        self.sparse_matrix = sparse_matrix
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self): return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        # 转换为 float32
        row_data = self.sparse_matrix[idx].toarray().squeeze().astype(np.float32)
        label = self.labels[idx] if self.labels is not None else torch.tensor(-1)
        return torch.from_numpy(row_data), label


class WideMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate):
        super(WideMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x): return self.layers(x)


def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================
# 2. 单模型训练器 (返回 验证集预测 和 测试集预测)
# ==========================================
def train_and_predict(config, train_df, val_df, test_df):
    model_name = config['name']
    print(f"[{model_name}] 启动... (Device: {config['device']})")

    # 1. 特征工程
    vectorizer = TfidfVectorizer(
        max_features=config['max_features'],
        ngram_range=config['ngram_range'],
        min_df=3
    )
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_val = vectorizer.transform(val_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])

    # 2. DataLoader
    train_ds = SparseRTDataset(X_train, train_df['Sentiment'].values)
    val_ds = SparseRTDataset(X_val, val_df['Sentiment'].values)  # 验证集用于 Stacking 训练
    test_ds = SparseRTDataset(X_test)  # 测试集用于最终预测

    # num_workers=0 保证线程安全
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # 3. 模型初始化
    torch.manual_seed(config['seed'])
    model = WideMLP(X_train.shape[1], config['hidden_dim'], 5, config['dropout']).to(config['device'])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()

    # 4. 训练
    for epoch in range(config['epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(config['device'], non_blocking=True)
            batch_y = batch_y.to(config['device'], non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    print(f"[{model_name}] 训练结束，正在生成预测特征...")

    # 5. 生成预测 (用于 Stacking)
    # 我们需要返回 Softmax 概率，而不是类别
    def get_probs(loader):
        model.eval()
        probs_list = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(config['device'])
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probs_list.append(probs)
        return np.concatenate(probs_list, axis=0)

    val_probs = get_probs(val_loader)  # 形状 (Val_Size, 5)
    test_probs = get_probs(test_loader)  # 形状 (Test_Size, 5)

    # 清理
    del model, optimizer, vectorizer, X_train
    if config['device'].type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return val_probs, test_probs


# ==========================================
# 3. 多线程管理类
# ==========================================
def worker_thread(configs, train_df, val_df, test_df, result_queue):
    """一个线程处理一列表的配置"""
    for cfg in configs:
        try:
            val_probs, test_probs = train_and_predict(cfg, train_df, val_df, test_df)
            result_queue.put((cfg['name'], val_probs, test_probs))
        except Exception as e:
            print(f"!!! [{cfg['name']}] 发生错误: {e}")


# ==========================================
# 4. 主程序：Stacking 流程
# ==========================================
if __name__ == "__main__":
    start_global = time.time()

    # --- A. 数据准备 ---
    print(">>> 1. 加载数据...")
    train_df_full = pd.read_csv('train.tsv', sep='\t')
    test_df = pd.read_csv('test.tsv', sep='\t')

    train_df_full['Phrase'] = train_df_full['Phrase'].fillna("")
    test_df['Phrase'] = test_df['Phrase'].fillna("")

    train_df_full['clean_text'] = train_df_full['Phrase'].apply(clean_text)
    test_df['clean_text'] = test_df['Phrase'].apply(clean_text)

    # 划分 训练集(用于训练基模型) 和 验证集(用于训练Stacking Meta模型)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=999)
    train_idx, val_idx = next(splitter.split(train_df_full, groups=train_df_full['SentenceId']))

    train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_df_full.iloc[val_idx].reset_index(drop=True)
    y_val = val_df['Sentiment'].values  # Stacking 的训练标签

    print(f"    训练集: {len(train_df)}, 验证集(Stacking用): {len(val_df)}")

    # --- B. 定义模型配置 ---
    base_cfg = {'epochs': 5, 'dropout': 0.5, 'lr': 1e-3}

    # GPU 任务列表 (Trigram, 80k features)
    gpu_configs = [
        {**base_cfg, 'name': 'GPU_Tri_1', 'device': torch.device('cuda'), 'ngram_range': (1, 3), 'max_features': 80000,
         'hidden_dim': 1024, 'batch_size': 128, 'seed': 101},
        {**base_cfg, 'name': 'GPU_Tri_2', 'device': torch.device('cuda'), 'ngram_range': (1, 3), 'max_features': 80000,
         'hidden_dim': 1024, 'batch_size': 128, 'seed': 102}
    ]

    # CPU 任务列表 (Bigram, 30k features)
    # CPU 训练通常较慢，所以把 batch_size 调小一点或者保持 64
    cpu_configs = [
        {**base_cfg, 'name': 'CPU_Bi_1', 'device': torch.device('cpu'), 'ngram_range': (1, 2), 'max_features': 30000,
         'hidden_dim': 512, 'batch_size': 64, 'seed': 201},
        {**base_cfg, 'name': 'CPU_Bi_2', 'device': torch.device('cpu'), 'ngram_range': (1, 2), 'max_features': 30000,
         'hidden_dim': 512, 'batch_size': 64, 'seed': 202}
    ]

    # --- C. 并行训练 (Efficiency Hack) ---
    print("\n>>> 2. 启动并行训练 (CPU线程 + GPU线程)...")
    result_queue = Queue()

    # 创建两个线程
    t_gpu = threading.Thread(target=worker_thread, args=(gpu_configs, train_df, val_df, test_df, result_queue))
    t_cpu = threading.Thread(target=worker_thread, args=(cpu_configs, train_df, val_df, test_df, result_queue))

    t_gpu.start()
    t_cpu.start()

    t_gpu.join()  # 等待 GPU 任务完成
    t_cpu.join()  # 等待 CPU 任务完成

    print(f">>> 基模型训练完成! 耗时: {time.time() - start_global:.1f}s")

    # --- D. 收集结果并准备 Stacking ---
    # 我们需要构建 Meta-Model 的输入特征
    # 这里的特征是：每个基模型输出的 5 个概率值
    # 总特征数 = 4个模型 * 5个类别 = 20维

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # 按名称排序，保证顺序一致
    results.sort(key=lambda x: x[0])

    val_meta_features = []
    test_meta_features = []

    print("\n>>> 3. Stacking 特征堆叠...")
    for name, v_prob, t_prob in results:
        print(f"    集成: {name}")
        val_meta_features.append(v_prob)
        test_meta_features.append(t_prob)

    # 拼接 (Samples, 20)
    X_meta_train = np.hstack(val_meta_features)
    X_meta_test = np.hstack(test_meta_features)

    # --- E. 训练 Meta-Learner (Logistic Regression) ---
    print("\n>>> 4. 训练 Meta-Learner (裁判模型)...")
    # 使用 LogisticRegression 寻找最佳权重组合
    meta_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    meta_model.fit(X_meta_train, y_val)

    # 验证 Meta-Model 的准确率
    val_preds = meta_model.predict(X_meta_train)
    acc = np.mean(val_preds == y_val)
    print(f"    >>> Stacking 验证集准确率: {acc:.4f} (通常高于单个模型)")

    # --- F. 最终预测 ---
    print(">>> 5. 生成最终预测...")
    final_predictions = meta_model.predict(X_meta_test)

    submission = pd.DataFrame({
        'PhraseId': test_df['PhraseId'],
        'Sentiment': final_predictions
    })

    submission.to_csv('submission_stacking.csv', index=False)
    print(">>> 完成！结果已保存至 'submission_stacking.csv'")