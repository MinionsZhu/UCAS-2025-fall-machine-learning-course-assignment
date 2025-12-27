import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
import re
import os
import time
import gc


# ==========================================
# 1. 基础类定义 (数据集与模型)
# ==========================================

class SparseRTDataset(Dataset):
    """
    内存优化数据集：保持数据为稀疏矩阵，仅在训练时转为Tensor。
    防止一次性转换撑爆内存。
    """

    def __init__(self, sparse_matrix, labels=None):
        self.sparse_matrix = sparse_matrix
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        # 转换为 float32 以兼容 PyTorch
        row_data = self.sparse_matrix[idx].toarray().squeeze().astype(np.float32)
        # 如果没有标签（测试集），返回占位符 -1
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

    def forward(self, x):
        return self.layers(x)


def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================
# 2. 核心训练函数 (支持不同配置)
# ==========================================

def train_single_model(config, train_df, val_df, test_df, model_name):
    print(f"\n[{model_name}] 正在初始化...")
    print(f"   |-- 设备: {config['device']}")
    print(f"   |-- N-gram: {config['ngram_range']}")
    print(f"   |-- Features: {config['max_features']}")

    # 设置随机种子
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # A. 特征工程 (独立生成，因为不同模型配置不同)
    print(f"[{model_name}] 生成 TF-IDF 特征...")
    vectorizer = TfidfVectorizer(
        max_features=config['max_features'],
        ngram_range=config['ngram_range'],
        min_df=3,
        stop_words=None
    )

    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_val = vectorizer.transform(val_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])

    # B. 准备 DataLoader
    train_ds = SparseRTDataset(X_train, train_df['Sentiment'].values)
    val_ds = SparseRTDataset(X_val, val_df['Sentiment'].values)

    # 如果是 GPU 训练，开启 pin_memory 加速传输
    use_pin = (config['device'].type == 'cuda')
    # num_workers=0 在 Windows 上最稳定
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, pin_memory=use_pin,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, pin_memory=use_pin, num_workers=0)

    # C. 初始化模型
    model = WideMLP(input_dim=X_train.shape[1],
                    hidden_dim=config['hidden_dim'],
                    num_classes=5,
                    dropout_rate=config['dropout']).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # D. 训练循环
    print(f"[{model_name}] 开始训练 ({config['epochs']} Epochs)...")
    model.train()  # 确保在训练模式

    for epoch in range(config['epochs']):
        for batch_X, batch_y in train_loader:
            # 显存优化关键：非阻塞传输
            batch_X = batch_X.to(config['device'], non_blocking=True)
            batch_y = batch_y.to(config['device'], non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # E. 验证集评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(config['device'])
            batch_y = batch_y.to(config['device'])
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    val_acc = correct / total
    print(f"[{model_name}] 训练完成! 验证集准确率: {val_acc:.4f}")

    # F. 生成测试集概率 (用于集成)
    print(f"[{model_name}] 生成预测概率...")
    test_ds = SparseRTDataset(X_test)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, pin_memory=use_pin, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(config['device'])
            outputs = model(batch_X)
            # Softmax 获取概率分布
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

    # G. 清理内存/显存 (非常重要)
    del model, optimizer, train_loader, val_loader, X_train, X_val, X_test, vectorizer
    if config['device'].type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return np.concatenate(all_probs, axis=0)


# ==========================================
# 3. 主程序：集成流程
# ==========================================

if __name__ == "__main__":
    if not os.path.exists('train.tsv'):
        print("未找到 train.tsv，请检查路径。")
        exit()

    # --- 1. 数据准备 ---
    print(">>> [全局] 加载与清洗数据...")
    train_df_full = pd.read_csv('train.tsv', sep='\t')
    test_df = pd.read_csv('test.tsv', sep='\t')

    train_df_full['Phrase'] = train_df_full['Phrase'].fillna("")
    test_df['Phrase'] = test_df['Phrase'].fillna("")

    train_df_full['clean_text'] = train_df_full['Phrase'].apply(clean_text)
    test_df['clean_text'] = test_df['Phrase'].apply(clean_text)

    # 固定划分，确保验证集一致
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=999)
    train_idx, val_idx = next(splitter.split(train_df_full, groups=train_df_full['SentenceId']))
    train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

    # --- 2. 配置列表 (已调整顺序：GPU 优先) ---
    common_cfg = {'epochs': 5, 'dropout': 0.5, 'learning_rate': 1e-3}

    configs = [
        # === 第一组：GPU 高性能模型 (优先运行) ===
        # 利用 4060 的 8G 显存，使用更大的 n-gram 和特征数
        {
            **common_cfg,
            'name': 'GPU_Model_1 (Tri-gram)',
            'device': torch.device('cuda'),
            'ngram_range': (1, 3),  # 捕捉长距离特征 "not very good"
            'max_features': 80000,  # 更大的词表
            'hidden_dim': 1024,  # 更宽的网络
            'batch_size': 128,  # GPU 适合大 Batch
            'seed': 2023
        },
        {
            **common_cfg,
            'name': 'GPU_Model_2 (Tri-gram)',
            'device': torch.device('cuda'),
            'ngram_range': (1, 3),
            'max_features': 80000,
            'hidden_dim': 1024,
            'batch_size': 128,
            'seed': 2025
        },

        # === 第二组：CPU 基础模型 (作为补充) ===
        # 特征较小，保证 CPU 训练不会太慢
        {
            **common_cfg,
            'name': 'CPU_Model_1 (Bi-gram)',
            'device': torch.device('cpu'),
            'ngram_range': (1, 2),
            'max_features': 30000,
            'hidden_dim': 512,
            'batch_size': 64,
            'seed': 42
        },
        {
            **common_cfg,
            'name': 'CPU_Model_2 (Bi-gram)',
            'device': torch.device('cpu'),
            'ngram_range': (1, 2),
            'max_features': 30000,
            'hidden_dim': 512,
            'batch_size': 64,
            'seed': 100
        }
    ]

    # --- 3. 逐个训练 ---
    ensemble_probs = []

    print(f"\n>>> 开始集成训练 (共 {len(configs)} 个模型, GPU 优先)...")
    start_time = time.time()

    for cfg in configs:
        # 检查 GPU 是否可用
        if cfg['device'].type == 'cuda' and not torch.cuda.is_available():
            print(f"警告: 系统未检测到 GPU，将 {cfg['name']} 降级为 CPU 运行（可能会很慢）。")
            cfg['device'] = torch.device('cpu')

        probs = train_single_model(cfg, train_df, val_df, test_df, cfg['name'])
        ensemble_probs.append(probs)

    print(f"\n>>> 所有模型训练完成！总耗时: {time.time() - start_time:.1f} 秒")

    # --- 4. 集成 (Soft Voting) ---
    print(">>> 正在进行概率平均 (Bagging)...")

    # 对 4 个模型的概率分布取平均
    avg_probs = np.mean(ensemble_probs, axis=0)

    # 取最大概率对应的类别
    final_predictions = np.argmax(avg_probs, axis=1)

    # --- 5. 保存 ---
    submission = pd.DataFrame({
        'PhraseId': test_df['PhraseId'],
        'Sentiment': final_predictions
    })

    filename = 'submission_ensemble_gpu_first.csv'
    submission.to_csv(filename, index=False)
    print(f">>> 结果已保存至 '{filename}'")