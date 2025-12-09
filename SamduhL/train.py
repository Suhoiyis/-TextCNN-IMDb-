import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from datasets import load_dataset, DatasetDict
import os
from tqdm import tqdm
import torch.nn.functional as F

# =========================================================================================
# 步骤一：定义模型
# =========================================================================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        # 告诉嵌入层哪个索引是填充符，训练时忽略它
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text: [batch size, seq len]
        embedded = self.embedding(text).unsqueeze(1) # [batch size, 1, seq len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs] # conved[n]: [batch size, num filters, seq len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved] # pooled[n]: [batch size, num filters]
        cat = torch.cat(pooled, dim=1) # [batch size, num filters * len(filter_sizes)]
        dropped = self.dropout(cat)
        return self.fc(dropped) # [batch size, output dim]

# =========================================================================================
# 步骤二：数据加载和预处理 (优化版)
# =========================================================================================
def load_and_preprocess_data(fixed_length=512):
    print("Loading and preprocessing data from Hugging Face Hub...")

    try:
        imdb_dataset = load_dataset("imdb")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return None, None, None, None

    print("Step 1: Tokenizing text...")
    tokenizer = lambda text: text.split()
    tokenized_datasets = imdb_dataset.map(lambda x: {'tokens': [tokenizer(text) for text in x['text']]}, batched=True)

    print("Step 2: Building vocabulary...")
    vocab = Counter(token for tokens in tokenized_datasets['train']['tokens'] for token in tokens)
    vocab = vocab.most_common(25000)
    word_to_idx = {word: i+2 for i, (word, _) in enumerate(vocab)}
    PAD_IDX = 0
    UNK_IDX = 1
    word_to_idx['<pad>'] = PAD_IDX
    word_to_idx['<unk>'] = UNK_IDX
    VOCAB_SIZE = len(word_to_idx)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    print(f"Step 3: Numericalizing and Padding to fixed length {fixed_length}...")

    def numericalize_and_pad(examples):
        all_token_indices = []
        for tokens in examples['tokens']:
            indices = [word_to_idx.get(token, UNK_IDX) for token in tokens]
            if len(indices) > fixed_length:
                indices = indices[:fixed_length]
            else:
                indices = indices + [PAD_IDX] * (fixed_length - len(indices))
            all_token_indices.append(indices)
        return {'input_ids': all_token_indices}

    processed_datasets = tokenized_datasets.map(
        numericalize_and_pad,
        batched=True, remove_columns=['text', 'tokens']
    )

    processed_datasets.set_format("torch", columns=["input_ids", "label"])

    return processed_datasets, word_to_idx, VOCAB_SIZE, PAD_IDX

# =========================================================================================
# 步骤三：定义训练和评估逻辑
# =========================================================================================
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss, epoch_acc = 0, 0
    model.train()
    for batch in tqdm(iterator, desc="Training"):
        optimizer.zero_grad()
        text = batch['input_ids']
        labels = batch['label']
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels.float())
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss, epoch_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            text = batch['input_ids']
            labels = batch['label']
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels.float())
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# =========================================================================================
# 步骤四：主程序 (包含模型保存)
# =========================================================================================
if __name__ == '__main__':
    # --- 超参数设置 ---
    NUM_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EMBEDDING_DIM = 100
    NUM_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    FIXED_MAX_LENGTH = 512

    # --- 设备设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"======== Using device: {device} ========")

    # --- 数据加载和预处理 ---
    print("======== Loading and preprocessing data... ========")
    processed_datasets, word_to_idx, VOCAB_SIZE, PAD_IDX = load_and_preprocess_data(fixed_length=FIXED_MAX_LENGTH)

    if processed_datasets is None:
        print("Failed to load data (check network connection). Exiting.")
        exit()

    # --- 创建 DataLoader ---
    def collate_batch_optimized(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch]).to(device, non_blocking=True)
        # 确保标签是 LongTensor 类型，以匹配 PyTorch BCEWithLogitsLoss 的期望（即使它是 float 也能工作，但这更规范）
        labels = torch.stack([item['label'] for item in batch]).to(device, non_blocking=True).long()
        return {'input_ids': input_ids, 'label': labels}

    # 使用 num_workers=0 避免多进程 CUDA 问题
    train_dataloader = DataLoader(processed_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch_optimized, num_workers=0)
    test_dataloader = DataLoader(processed_datasets['test'], batch_size=BATCH_SIZE, collate_fn=collate_batch_optimized, num_workers=0)
    print("======== DataLoaders created (Optimized). ========")

    # --- 模型、优化器、损失函数 ---
    model = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss().to(device)
    print("======== Model, Optimizer, and Criterion initialized. ========")

    # --- 训练循环 ---
    print("======== Starting training... ========")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_dataloader, criterion)

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

    print("======== Training finished. ========")

    # --- 保存最终的模型 ---
    MODEL_SAVE_PATH = "textcnn_model.pth"
    print(f"\n======== Saving final model to {MODEL_SAVE_PATH} ========")
    # 只保存模型学习到的参数（state_dict）
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("======== Model saved successfully. ========")