import sys
sys.path.append("../")
import vec2text
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR
from torch_geometric.loader import NeighborLoader

import gnn
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
import vec2text.models.inversion as inversion
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import random
import numpy as np
import gzip
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from analyze_utils import load_experiment_and_trainer
from transformers import AdamW
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch import Tensor

device = 'cuda:3'
# model_path = "/data/whuak/geia_graph/vec2text/vec2text/saves/autoencoder_trainable_title_abstract_contrasive/checkpoint-168000"
log_file_path = 'autoencoder_from_scratch_256.log'

lr = 0.0001 # 0.0001
num_epochs = 100
eval_every = 1000
batch_size = 16
num_warmup_steps = 1000
is_encoder_trainable = True
num_sampled_neighbour = 5
max_seq_len = 256

def encoder_train(model, data_x, target_train, optimizer, criterion, scheduler=None):
    model.train()
    optimizer.zero_grad()
    out = model(data_x)
    loss = criterion(out, target_train)
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()
    return loss.item()


@torch.no_grad()
def encoder_test(model, data_x, target_test):
    model.eval()
    out = model(data_x)
    y_pred = out.argmax(dim=-1, keepdim=True).reshape(-1,)
    correct_count = ((y_pred == target_test).sum().item()) 
    return correct_count


def get_warmup_scheduler(optimizer, num_warmup_steps):    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 1.0

    # Create the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return lr_scheduler

logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

dataset=PygNodePropPredDataset(name='ogbn-arxiv', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
data = dataset[0]
data = data.to(device)

split_idx = dataset.get_idx_split() # train:test:valid = 90941/48603/29799
train_split = split_idx['train'].to(device)
test_split = split_idx['test'].to(device)

print('loading the graph meta data...')
df_node2paper = pd.read_csv('dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
file_path = 'titleabs.tsv.gz'
df_paperid2titleabs = pd.read_csv(gzip.open(file_path, 'rt'), delimiter='\t', names=['paperid', 'title', 'abstract'])
df_paperid2titleabs = df_paperid2titleabs.drop(0)
df_paperid2titleabs = df_paperid2titleabs.drop(179719)
def fn(x):
    return int(x['paperid'])
df_paperid2titleabs['paperid_int'] = df_paperid2titleabs.apply(fn, axis=1)

print('building adjacency list...')
adj_list = {}
sparse_tensor = data.adj_t
for row in range(sparse_tensor.size(0)):
    neighbors = []
    start = sparse_tensor.crow_indices()[row]
    end = sparse_tensor.crow_indices()[row + 1]
    for idx in range(start, end):
        neighbors.append(sparse_tensor.col_indices()[idx].item())
    adj_list[row] = neighbors

# experiment, trainer = load_experiment_and_trainer(model_path)

embedder_encoder = T5ForConditionalGeneration.from_pretrained('sentence-transformers/sentence-t5-base')
embedder_encoder = embedder_encoder.to(device)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class encoder(nn.Module):
    def __init__(self,):
        super(encoder, self).__init__()
        self.embedder_encoder = embedder_encoder
        self.sage_layer = nn.Sequential(nn.Linear(768*2, 1024))
        self.cls_layer = nn.Linear(1024, 40)
        self.bn1 = nn.BatchNorm1d(1024)
        self.tokenizer_t5 = T5Tokenizer.from_pretrained('sentence-transformers/sentence-t5-base')

        
    def forward(self, node_idx):
        # Forward pass
        sent_emb_list = []
        for idx_ in node_idx:
            title_list = []
            idx_int = idx_.item()
            sampled_neighbours_ids = random.sample(adj_list[idx_int], min(num_sampled_neighbour, len(adj_list[idx_int])))
            for n_id_ in sampled_neighbours_ids:
                paperid = df_node2paper[df_node2paper['node idx'] == n_id_]['paper id'].values[0]
                title = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['title'].values[0]
                abstract = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['abstract'].values[0]
                title_list.append('title: ' + title + '; abstract: ' + abstract)
            # tokenized_title = self.embedder_tokenizer(title_list, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
            # model_output = self.embedder({"input_ids": tokenized_title})
            # sent_embs = model_output["sentence_embedding"] 
            input_ids_title_list = self.tokenizer_t5(title_list, padding=True, truncation=True, return_tensors="pt" ,max_length=max_seq_len).to(device)
            # sent_embs = self.embedder.encode(title_list, convert_to_tensor=True, device=device) # (#neighbour, 768)
            sent_embs_last_state = self.embedder_encoder.encoder(input_ids=input_ids_title_list['input_ids'], attention_mask=input_ids_title_list['attention_mask']).last_hidden_state # (#neighbour, 768)
            sent_embs = average_pool(sent_embs_last_state, input_ids_title_list['attention_mask'])
            sent_embs_pooling = sent_embs.mean(axis=0)
            sent_emb_list.append(sent_embs_pooling)

        neighbour_sent_embs_batch = torch.stack(sent_emb_list, dim=0) # (batch size, 768)
        
        title_list = []
        for idx_ in node_idx:
            idx_int = idx_.item()
            paperid = df_node2paper[df_node2paper['node idx'] == idx_int]['paper id'].values[0]
            title = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['title'].values[0]
            abstract = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['abstract'].values[0]
            title_list.append('title: ' + title + '; abstract: ' + abstract)
        input_ids_title_list = self.tokenizer_t5(title_list, padding=True, truncation=True, return_tensors="pt" ,max_length=max_seq_len).to(device)
        center_sent_embs_batch_last_hidden_state = self.embedder_encoder.encoder(input_ids=input_ids_title_list['input_ids'], attention_mask=input_ids_title_list['attention_mask']).last_hidden_state
        center_sent_embs_batch = average_pool(center_sent_embs_batch_last_hidden_state, input_ids_title_list['attention_mask'])
        # center_sent_embs_batch = center_sent_embs_batch.mean(axis=1)

        embeddings = torch.concatenate([center_sent_embs_batch, neighbour_sent_embs_batch], dim=1)
        
        graph_emb = self.sage_layer(embeddings)

        graph_emb = self.bn1(graph_emb)
        graph_emb = F.relu(graph_emb)

        out = self.cls_layer(graph_emb)

        return out
    
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

dataset = MyDataset(train_split, data.y[train_split].reshape(-1,))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataset = MyDataset(test_split, data.y[test_split].reshape(-1,))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = encoder()
model = model.to(device)

if not is_encoder_trainable:
    for param in model.embedder.parameters():
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_warmup_scheduler(optimizer, num_warmup_steps)

itr = 0

# exit(0)
for epoch in range(num_epochs):
    # Training loop
    total_loss = 0
    total_acc = 0
    for batch in tqdm(dataloader):
        itr += 1
        model.train()
        input_idx, labels = batch
        loss_item = encoder_train(model, input_idx, labels, optimizer, criterion, scheduler)
        logging.info(f'iteration {itr} loss: {loss_item}, lr: {scheduler.get_last_lr()[0]}')
        if not itr % eval_every:
            with torch.no_grad():
                correct_count, total = 0, 0
                for test_batch in tqdm(test_dataloader):
                    input_idx, labels = test_batch
                    correct_count += encoder_test(model, input_idx, labels)
                    total += len(labels)
                acc = correct_count / total
                logging.info(f'accuracy after iteration {itr}: {acc}')


