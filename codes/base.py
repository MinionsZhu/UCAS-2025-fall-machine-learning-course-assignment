import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
import re
import os

# ==========================================
# 1. 配置参数
# ==========================================
CONFIG = {
    'max_features': 30000,  # 词表大小
    'ngram_range': (1, 2),  # 关键: 使用Bigram捕捉 "not good" 等反转语义
    'hidden_dim': 512,  # Wide MLP: 宽隐藏层
    'dropout': 0.5,  # 强正则化
    'batch_size': 64,
    'learning_rate': 1e-3,
    'epochs': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}


def clean_text(text):
    """
    轻量级数据清洗. 仅保留字母, 数字和基本标点, 移除多余空白. 不移除 "not", "very", "but".
    """
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\']', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================
# 2. 数据加载与处理
# ==========================================
def load_and_process_data():
    print(">>> 正在加载数据...")
    # 读取 tsv 格式数据并清洗
    train_df = pd.read_csv('train.tsv', sep='\t')
    test_df = pd.read_csv('test.tsv', sep='\t')

    train_df['Phrase'] = train_df['Phrase'].fillna("")
    test_df['Phrase'] = test_df['Phrase'].fillna("")

    print(">>> 正在清洗文本...")
    train_df['clean_text'] = train_df['Phrase'].apply(clean_text)
    test_df['clean_text'] = test_df['Phrase'].apply(clean_text)

    # 划分验证集. 使用 SentenceId 作为 group id, 因为 train.tsv 中同一句子的不同部分不能分属训练集和验证集.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(splitter.split(train_df, groups=train_df['SentenceId']))

    train_subset = train_df.iloc[train_idx].reset_index(drop=True)
    val_subset = train_df.iloc[val_idx].reset_index(drop=True)

    print(f"训练集大小: {len(train_subset)}, 验证集大小: {len(val_subset)}")

    # TF-IDF 向量化 (BoW)
    print(">>> 正在生成 TF-IDF 特征 (Bag-of-Words)...")
    vectorizer = TfidfVectorizer(
        max_features=CONFIG['max_features'],
        ngram_range=CONFIG['ngram_range'],
        min_df=3,
        stop_words=None
    )

    # 将清洗后的文本转换为 TF-IDF 特征矩阵, 列即为词表, 行为样本
    X_train = vectorizer.fit_transform(train_subset['clean_text'])
    X_val = vectorizer.transform(val_subset['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])

    print(">>> 转换数据为 Tensor...")

    # 将稀疏矩阵转为 Tensor
    def to_tensor(sparse_matrix):
        return torch.FloatTensor(sparse_matrix.toarray())

    # 转换为 PyTorch 数据集
    train_dataset = RTDataset(to_tensor(X_train), train_subset['Sentiment'].values)
    val_dataset = RTDataset(to_tensor(X_val), val_subset['Sentiment'].values)
    # 测试集没有标签, 仅仅做特征矩阵的转换
    test_tensor = to_tensor(X_test)

    return train_dataset, val_dataset, test_tensor, test_df['PhraseId']


# ==========================================
# 3. 简单的 PyTorch 数据集类
# ==========================================
class RTDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


# ==========================================
# 4. 模型中的宽层 MLP 定义
# ==========================================
class WideMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(WideMLP, self).__init__()

        # 结构: Input -> 2*(Linear -> BatchNorm -> ReLU -> Dropout) -> Linear -> Output
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 额外加一层以增强非线性能力
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


# ==========================================
# 5. 训练与评估
# ==========================================
def train_model(train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    input_dim = train_dataset.features.shape[1]
    model = WideMLP(input_dim, CONFIG['hidden_dim'], num_classes=5).to(CONFIG['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

    print(f">>> 开始训练 (Device: {CONFIG['device']})...")

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(CONFIG['device']), batch_y.to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证
        val_acc = evaluate(model, val_loader)
        print(
            f"Epoch {epoch + 1}/{CONFIG['epochs']} | Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    return model


def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(CONFIG['device'])
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    return accuracy_score(all_labels, all_preds)


def generate_submission(model, test_tensor, phrase_ids):
    print(">>> 正在生成预测结果...")
    model.eval()
    test_loader = DataLoader(RTDataset(test_tensor), batch_size=CONFIG['batch_size'], shuffle=False)
    predictions = []

    with torch.no_grad():
        for batch_X in test_loader:
            batch_X = batch_X.to(CONFIG['device'])
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    submission = pd.DataFrame({
        'PhraseId': phrase_ids,
        'Sentiment': predictions
    })

    submission.to_csv('submission.csv', index=False)
    print(">>> 任务完成！结果已保存至 'submission.csv'")


# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 检查文件是否存在
    if not os.path.exists('train.tsv') or not os.path.exists('test.tsv'):
        print("错误: 当前目录下未找到 train.tsv 或 test.tsv.")
        print("请下载 Rotten Tomatoes 数据集并放置在当前目录.")
    else:
        # 1. 加载数据
        train_ds, val_ds, test_tensor, test_ids = load_and_process_data()

        # 2. 训练模型
        model = train_model(train_ds, val_ds)

        # 3. 生成提交文件
        generate_submission(model, test_tensor, test_ids)
