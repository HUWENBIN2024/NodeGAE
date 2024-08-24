import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from simteg.src.dataset import load_dataset
import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from tqdm import tqdm
import logging
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW

import torch
from torch.utils.data import Dataset
import os
import argparse
import pandas as pd 
import os.path as osp


parser = argparse.ArgumentParser(description='OGBN-Products (GNN)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
# parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_size_gnn', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay_factor', type=float, default=1e-5)
parser.add_argument('--warm_up_ratio', type=float, default=0.15)

parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=8)


parser.add_argument('--model_name', type=str, default='sage')

parser.add_argument('--label_smoothing', type=float, default=0) # 0.3
parser.add_argument('--weight_decay', type=float, default=0) # 4e-6
parser.add_argument('--log_path', type=str, default='logs/lm_finetune_t5base_node_cls.logs')
parser.add_argument('--save_path', type=str, default='save_lm_finetune/t5base_mode_cls.pt')

args = parser.parse_args()
print(args)

logging.basicConfig(filename=args.log_path, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Load the dataset
device = args.device
lr = args.lr # 0.000001
eval_every = args.eval_every
batch_size = args.batch_size
num_epochs = args.num_epochs
weight_decay_factor = args.weight_decay_factor # 1e-5
warm_up_ratio = args.warm_up_ratio
label_smoothing = args.label_smoothing # 0.1
header_dropout_prob = 0 # 0.5

max_seq_len = 256


print('Loading the dataset...')
dataset_name = 'products'
dataset = load_dataset(name='ogbn-products', tokenizer='sentence-transformers/sentence-t5-base')
data = dataset._data

split_idx = dataset.get_idx_split()
train_split = split_idx['train'].to(device)
val_split = split_idx['valid'].to(device)
y_train = data.y[train_split].cpu().numpy()
y_val = data.y[val_split].cpu().numpy()
train_split_list = train_split.cpu().numpy().tolist()
val_split_list = val_split.cpu().numpy().tolist()


print('Loading the text data...')
text_dir = osp.join('/localdata/hjingaa/github/geia_graph/baseline_graph_lm/dataset/ogbn_products_text/raw/Amazon-3M.raw')
df = pd.read_csv('/localdata/hjingaa/github/geia_graph/baseline_graph_lm/dataset/ogbn_products_text/raw/Amazon-3M.raw/products.csv', sep=" ")
df.replace(np.nan, "", inplace=True)
df["titlecontent"] = df["title"] + ". " + df["content"]
df = df.drop(columns=["title", "content"])
df.rename(columns={"uid": "asin"}, inplace=True)

df_mapping = pd.read_csv("/localdata/hjingaa/github/geia_graph/baseline_graph_lm/dataset/ogbn_products_text/mapping/nodeidx2asin.csv.gz")
df = df_mapping.merge(df, how="left", on="asin")
text_list = df["titlecontent"].values.tolist()
df = pd.DataFrame(text_list)
df.reset_index(inplace=True)
df = df.rename(columns={0:'text'})
df_id2text = df

split_idx = dataset.get_idx_split()

train_split = split_idx['train'].to(device)
val_split = split_idx['valid'].to(device)
test_split = split_idx['test'].to(device)

y_train = data.y[train_split].cpu().numpy()
y_val = data.y[val_split].cpu().numpy()
y_test = data.y[test_split].cpu().numpy()

train_split_list = train_split.cpu().numpy().tolist()
val_split_list = val_split.cpu().numpy().tolist()
test_split_list = test_split.cpu().numpy().tolist()


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]



dataset_train = MyDataset(df_id2text.iloc[train_split_list]['text'].to_numpy(), y_train)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_val = MyDataset(df_id2text.iloc[val_split_list]['text'].to_numpy(), y_val)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

dataset_test = MyDataset(df_id2text.iloc[test_split_list]['text'].to_numpy(), y_test)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_warmup_scheduler(optimizer, num_warmup_steps):    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 1.0

    # Create the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return lr_scheduler

# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/sentence-t5-base')
t5_model = AutoModel.from_pretrained('sentence-transformers/sentence-t5-base')

class e5_cls_model(nn.Module):
    def __init__(self):
        super(e5_cls_model, self).__init__()
        self.t5_model = t5_model.encoder
        # Define layers here
        self.mlp = nn.Sequential( 
            nn.Linear(in_features=768, out_features=256), 
            nn.ReLU(),
            nn.Dropout(header_dropout_prob),
            nn.Linear(in_features=256, out_features=40),
        )   
    def forward(self, x):
        outputs = self.t5_model(**x)
        embeddings = average_pool(outputs.last_hidden_state, x['attention_mask'])
        logits = self.mlp(embeddings)
        return logits


# Tokenize the input texts

model  = e5_cls_model()
model.to(device)



def train_model():
    # Tokenize the input text
    # input_ids = [tokenizer.encode(text, padding='max_length', max_length=128, return_tensors='pt') for text in X]
    
    # Compute the loss and accuracy during training
    
    # dataset = TensorDataset(torch.cat(input_ids, dim=0), torch.tensor(y))
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # val_input_ids = [tokenizer.encode(text, padding='max_length', max_length=128, return_tensors='pt') for text in X_val]
    # val_dataset = TensorDataset(torch.cat(val_input_ids, dim=0), torch.tensor(y_val))
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_factor)
    total_step = len(dataloader_train) * num_epochs
    scheduler = get_warmup_scheduler(optimizer, int(total_step * warm_up_ratio))
    itr = 0
    best_val_acc = 0
    for epoch in range(num_epochs):
        # Training loop
        total_loss = 0
        total_acc = 0
        for batch in tqdm(dataloader_train):
            itr += 1
            model.train()
            text, labels = batch
            tokenized_text = tokenizer(text, padding='max_length', max_length=max_seq_len, return_tensors='pt', truncation=True)
            labels = torch.tensor(labels.reshape(-1,))
            tokenized_text, labels = tokenized_text.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(tokenized_text)
            loss = criterion(output, labels)
            # total_loss += loss.item()
            # acc = (output.argmax(dim=1) == labels).float().mean()
            # total_acc += acc.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            logging.info(f'iteration {itr+1}: train loss: {loss.item()}, lr: {scheduler.get_last_lr()[0]}')

            if not itr % eval_every:
                # Validation loop
                model.eval()
                val_total = 0
                val_correct = 0
                with torch.no_grad():
                    print('start evaluation')
                    for batch in tqdm(dataloader_test):
                        text_val, labels_val = batch
                        labels_val = torch.tensor(labels_val.reshape(-1,))
                        tokenized_text_val = tokenizer(text_val, padding='max_length', max_length=max_seq_len, return_tensors='pt', truncation=True)
                        tokenized_text_val, labels_val = tokenized_text_val.to(device), labels_val.to(device)
                        output_val = model(tokenized_text_val)
                        val_correct += (output_val.argmax(dim=1) == labels_val).sum().item()
                        val_total += len(labels_val)
                    val_acc = val_correct/val_total
                    logging.info(f'iteration {itr+1}: Val Acc: {val_acc}')
                    if best_val_acc < val_acc:
                        best_val_acc = val_acc
                        # if not os.path.exists('save_model'):
                        #     os.makedirs('save_model', exist_ok=True)
                        torch.save(model, args.save_path)

        # print(f'iteration {itr+1}: Train Loss: {total_loss/len(dataloader)}, Train Acc: {total_acc/len(dataloader)}, Val Loss: {val_loss/len(val_dataloader)}, Val Acc: {val_acc/len(val_dataloader)}')

# Train the model
train_model()

