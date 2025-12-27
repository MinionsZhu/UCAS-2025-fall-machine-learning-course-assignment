import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import re
import os
import time
import gc
import threading
from queue import Queue
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. 基础组件 (Dataset & Model) - (无变化)
# ==========================================
class SparseRTDataset(Dataset):
    def __init__(self, sparse_matrix, labels=None):
        self.sparse_matrix = sparse_matrix
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self): return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        row_data = self.sparse_matrix[idx].toarray().squeeze().astype(np.float32)
        label = self.labels[idx] if self.labels is not None else torch.tensor(-1)
        return torch.from_numpy(row_data), label


class WideMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate):
        super(WideMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
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
# 2. 单模型训练器 - (无变化)
# ==========================================
def train_and_predict(config, train_df, val_df, test_df):
    model_name = config['name']
    print(f"[{model_name}] 启动... (Device: {config['device']})")
    vectorizer = TfidfVectorizer(max_features=config['max_features'], ngram_range=config['ngram_range'], min_df=3)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_val = vectorizer.transform(val_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])
    train_ds = SparseRTDataset(X_train, train_df['Sentiment'].values)
    val_ds = SparseRTDataset(X_val, val_df['Sentiment'].values)
    test_ds = SparseRTDataset(X_test)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    torch.manual_seed(config['seed'])
    model = WideMLP(X_train.shape[1], config['hidden_dim'], 5, config['dropout']).to(config['device'])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config['epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(config['device'], non_blocking=True), batch_y.to(config['device'],
                                                                                           non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    print(f"[{model_name}] 训练结束，正在生成预测特征...")

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

    val_probs = get_probs(val_loader)
    test_probs = get_probs(test_loader)
    del model, optimizer, vectorizer, X_train
    if config['device'].type == 'cuda': torch.cuda.empty_cache()
    gc.collect()
    return val_probs, test_probs


# ==========================================
# 3. 多线程管理 - (无变化)
# ==========================================
def worker_thread(configs, train_df, val_df, test_df, result_queue):
    for cfg in configs:
        try:
            val_probs, test_probs = train_and_predict(cfg, train_df, val_df, test_df)
            result_queue.put((cfg['name'], val_probs, test_probs))
        except Exception as e:
            print(f"!!! [{cfg['name']}] 发生错误: {e}")


# ==========================================
# 4. 可视化函数 - (无变化)
# ==========================================
def plot_top_tf_idf_words(text, vectorizer):
    print(f"\n[BoW 可视化] 正在分析测试集中的句子: '{text}'")
    vector = vectorizer.transform([clean_text(text)])
    feature_names = vectorizer.get_feature_names_out()
    non_zero_indices = vector.nonzero()[1]
    scores = vector.data
    df = pd.DataFrame({'word': feature_names[non_zero_indices], 'tfidf': scores})
    df = df.sort_values(by='tfidf', ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='tfidf', y='word', data=df, palette='viridis')
    plt.title(f'Top 15 TF-IDF Features for a Real Test Sentence')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Word / Phrase')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Stacking Model Confusion Matrix (Normalized on Validation Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_model_correlation(val_meta_features, model_names):
    pred_df = pd.DataFrame()
    for i, name in enumerate(model_names):
        model_preds = np.argmax(val_meta_features[:, i * 5:(i + 1) * 5], axis=1)
        pred_df[name] = model_preds
    corr = pred_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('Correlation of Base Model Predictions on Validation Set')
    plt.show()


# ==========================================
# 5. 主程序 (加入可视化调用)
# ==========================================
if __name__ == "__main__":
    start_global = time.time()

    # --- A. 数据准备 ---
    print(">>> 1. 加载数据...")
    train_df_full = pd.read_csv('train.tsv', sep='\t')
    test_df = pd.read_csv('test.tsv', sep='\t')
    train_df_full['Phrase'] = train_df_full['Phrase'].fillna("")
    test_df['Phrase'] = test_df['Phrase'].fillna("")

    # ===【 BUG 修复 】===
    # 从 'Phrase' 列创建 'clean_text' 列
    train_df_full['clean_text'] = train_df_full['Phrase'].apply(clean_text)
    test_df['clean_text'] = test_df['Phrase'].apply(clean_text)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=999)
    train_idx, val_idx = next(splitter.split(train_df_full, groups=train_df_full['SentenceId']))
    train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_df_full.iloc[val_idx].reset_index(drop=True)
    y_val = val_df['Sentiment'].values

    # --- B. 定义模型配置 ---
    base_cfg = {'epochs': 5, 'dropout': 0.5, 'lr': 1e-3}
    gpu_configs = [
        {**base_cfg, 'name': 'GPU_Tri_1', 'device': torch.device('cuda'), 'ngram_range': (1, 3), 'max_features': 80000,
         'hidden_dim': 1024, 'batch_size': 128, 'seed': 101},
        {**base_cfg, 'name': 'GPU_Tri_2', 'device': torch.device('cuda'), 'ngram_range': (1, 3), 'max_features': 80000,
         'hidden_dim': 1024, 'batch_size': 128, 'seed': 102}
    ]
    cpu_configs = [
        {**base_cfg, 'name': 'CPU_Bi_1', 'device': torch.device('cpu'), 'ngram_range': (1, 2), 'max_features': 30000,
         'hidden_dim': 512, 'batch_size': 64, 'seed': 201},
        {**base_cfg, 'name': 'CPU_Bi_2', 'device': torch.device('cpu'), 'ngram_range': (1, 2), 'max_features': 30000,
         'hidden_dim': 512, 'batch_size': 64, 'seed': 202}
    ]

    # --- C & D & E (训练和Stacking) ---
    print("\n>>> 2. 启动并行训练...")
    result_queue = Queue()
    t_gpu = threading.Thread(target=worker_thread, args=(gpu_configs, train_df, val_df, test_df, result_queue))
    t_cpu = threading.Thread(target=worker_thread, args=(cpu_configs, train_df, val_df, test_df, result_queue))
    t_gpu.start()
    t_cpu.start()
    t_gpu.join()
    t_cpu.join()

    results = []
    while not result_queue.empty(): results.append(result_queue.get())
    results.sort(key=lambda x: x[0])

    val_meta_features = np.hstack([r[1] for r in results])
    test_meta_features = np.hstack([r[2] for r in results])

    print("\n>>> 4. 训练 Meta-Learner...")
    meta_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    meta_model.fit(val_meta_features, y_val)
    val_preds = meta_model.predict(val_meta_features)
    acc = np.mean(val_preds == y_val)
    print(f"    >>> Stacking 验证集准确率: {acc:.4f}")

    # --- F. 最终预测 & 保存 ---
    final_predictions = meta_model.predict(test_meta_features)
    submission = pd.DataFrame({'PhraseId': test_df['PhraseId'], 'Sentiment': final_predictions})
    submission.to_csv('submission_stacking.csv', index=False)
    print(">>> 结果已保存至 'submission_stacking.csv'")

    # --- G. 可视化分析 ---
    print("\n>>> G. 生成可视化图表...")

    vis_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=80000, min_df=3)
    vis_vectorizer.fit(train_df['clean_text'])

    test_df['phrase_len'] = test_df['Phrase'].str.len()
    sorted_test_by_len = test_df.sort_values(by='phrase_len', ascending=False)
    sample_text_from_test = sorted_test_by_len['Phrase'].iloc[20]

    plot_top_tf_idf_words(sample_text_from_test, vis_vectorizer)

    class_names = ['0-neg', '1-s-neg', '2-neu', '3-s-pos', '4-pos']
    plot_confusion_matrix(y_val, val_preds, class_names)

    model_names = [r[0] for r in results]
    plot_model_correlation(val_meta_features, model_names)

    print(">>> 可视化完成！")