import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
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
import gzip
import pandas as pd
from sklearn.metrics import roc_auc_score
import argparse
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
from src.classifier.link_prediction.utils import eval_hits, eval_mrr

parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
# parser.add_argument('--use_sage', action='store_true')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_size_gnn', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay_factor', type=float, default=1e-5)
parser.add_argument('--warm_up_ratio', type=float, default=0)

parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=8)


parser.add_argument('--model_name', type=str, default='sage')

parser.add_argument('--label_smoothing', type=float, default=0) # 0.3
parser.add_argument('--weight_decay', type=float, default=0) # 4e-6

parser.add_argument('--is_emb_from_path', type=bool, default=True)
parser.add_argument('--log_path', type=str, default='logs/lm_finetune_t5base.logs')
parser.add_argument('--save_path', type=str, default='save_lm_finetune/t5base_link_pred.pt')


args = parser.parse_args()
print(args)
os.makedirs('save_lm_finetune', exist_ok=True)
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename=args.log_path, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


# Load the dataset
device = args.device
lr = args.lr # 0.000001
eval_every = args.eval_every
batch_size = args.batch_size
num_epochs = args.num_epochs
weight_decay_factor = args.weight_decay_factor # 1e-5
warm_up_ratio = args.warm_up_ratio # 0.15
# label_smoothing = 0 # 0.1
header_dropout_prob = 0 # 0.5

max_seq_len = 256

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

# graph_dataset=PygNodePropPredDataset(name='ogbn-arxiv', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
# data = graph_dataset[0]
# data = data.to(device)


train_data = torch.load('datasets/ogbn_arxiv/split/link_pred/train').to(device)
test_data = torch.load('datasets/ogbn_arxiv/split/link_pred/test').to(device)
val_data = torch.load('datasets/ogbn_arxiv/split/link_pred/val').to(device)


print('loading the graph meta data...')
df_node2paper = pd.read_csv('datasets/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')
file_path = 'datasets/ogbn_arxiv/titleabs.tsv.gz'
df_paperid2titleabs = pd.read_csv(gzip.open(file_path, 'rt'), delimiter='\t', names=['paperid', 'title', 'abstract'])
df_paperid2titleabs = df_paperid2titleabs.drop(0)
df_paperid2titleabs = df_paperid2titleabs.drop(179719)
def fn(x):
    return int(x['paperid'])
df_paperid2titleabs['paperid_int'] = df_paperid2titleabs.apply(fn, axis=1)


# split_idx = graph_dataset.get_idx_split() # train:test:valid = 90941/48603/29799
# train_split = split_idx['train'].to(device)
# test_split = split_idx['test'].to(device)


# y_train = data.y[train_split]  # Replace with your actual label array
# y_val = data.y[test_split]



# dataset_train = MyDataset(arxiv_data['train']['text'], y_train)
# dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
# dataset_test = MyDataset(arxiv_data['test']['text'], y_val)
# dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_warmup_scheduler(optimizer, num_warmup_steps):    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps and num_warmup_steps >0:
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

class t5_link_pred_model(nn.Module):
    def __init__(self):
        super(t5_link_pred_model, self).__init__()
        self.t5_model = t5_model.encoder
        # Define layers here
        
    def forward(self, x):
        outputs = self.t5_model(**x)
        embeddings = average_pool(outputs.last_hidden_state, x['attention_mask'])
        return embeddings
    
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

# Tokenize the input texts

model  = t5_link_pred_model()
model.to(device)

predictor = LinkPredictor(768, 768, 1, 3, 0).to(device)
predictor.to(device)

# def get_tokenized_data(src, tar, tar_neg):
#     title_abs_list_src = []
#     title_abs_list_tar_pos = []
#     title_abs_list_tar_neg = []
    
#     for i in range(len(src)):
#         idx_src_ = src[i].item()
#         paperid_src = df_node2paper[df_node2paper['node idx'] == idx_src_]['paper id'].values[0]
#         title_src = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid_src]['title'].values[0]
#         abstract_src = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid_src]['abstract'].values[0]

#         idx_tar_ = tar[i].item()
#         paperid_tar = df_node2paper[df_node2paper['node idx'] == idx_tar_]['paper id'].values[0]
#         title_tar = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid_tar]['title'].values[0]
#         abstract_tar = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid_tar]['abstract'].values[0]

#         idx_tar_neg_ = tar_neg[i].item()
#         paperid_tar_neg = df_node2paper[df_node2paper['node idx'] == idx_tar_neg_]['paper id'].values[0]
#         title_tar_neg = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid_tar_neg]['title'].values[0]
#         abstract_tar_neg = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid_tar_neg]['abstract'].values[0]

#         title_abs_list_src.append('title: ' + title_src + '; abstract: ' + abstract_src)
#         title_abs_list_tar_pos.append('title: ' + title_tar + '; abstract: ' + abstract_tar)
#         title_abs_list_tar_neg.append('title: ' + title_tar_neg + '; abstract: ' + abstract_tar_neg)
        
#     tokenized_text_src = tokenizer(title_abs_list_src, padding=True, truncation=True, return_tensors="pt" ,max_length=max_seq_len).to(device)
#     tokenized_text_tar_pos = tokenizer(title_abs_list_tar_pos, padding=True, truncation=True, return_tensors="pt" ,max_length=max_seq_len).to(device)
#     tokenized_text_tar_neg = tokenizer(title_abs_list_tar_neg, padding=True, truncation=True, return_tensors="pt" ,max_length=max_seq_len).to(device)


#     return tokenized_text_src, tokenized_text_tar_pos, tokenized_text_tar_neg

def get_tokenized_data(src):
    title_abs_list_src = []
    
    for i in range(len(src)):
        idx_src_ = src[i].item()
        paperid_src = df_node2paper[df_node2paper['node idx'] == idx_src_]['paper id'].values[0]
        title_src = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid_src]['title'].values[0]
        abstract_src = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid_src]['abstract'].values[0]
        title_abs_list_src.append('title: ' + title_src + '; abstract: ' + abstract_src)
    tokenized_text_src = tokenizer(title_abs_list_src, padding=True, truncation=True, return_tensors="pt" ,max_length=max_seq_len).to(device)

    return tokenized_text_src

train_label_index_len = len(train_data.edge_label_index[0])
valid_label_index_len = len(val_data.edge_label_index[0])
test_label_index_len = len(test_data.edge_label_index[0])

# for evaluation, create negative samples
neg_train_edge = torch.randint(0, train_data.num_nodes, (train_label_index_len//2, 2), dtype=torch.long, device=train_data.x.device)
neg_valid_edge = torch.randint(0, val_data.num_nodes, (valid_label_index_len//2, 2), dtype=torch.long, device=train_data.x.device)
neg_test_edge = torch.randint(0, test_data.num_nodes, (test_label_index_len//2, 2), dtype=torch.long, device=train_data.x.device)

@torch.no_grad()
def evaluation():
    model.eval()
    train_source = train_data.edge_label_index[0,:train_label_index_len//2].to(train_data.x.device)
    train_target = train_data.edge_label_index[1,:train_label_index_len//2].to(train_data.x.device)
    
    test_source = test_data.edge_label_index[0,:test_label_index_len//2].to(test_data.x.device)
    test_target = test_data.edge_label_index[1,:test_label_index_len//2].to(test_data.x.device)

    valid_source = val_data.edge_label_index[0,:valid_label_index_len//2].to(val_data.x.device)
    valid_target = val_data.edge_label_index[1,:valid_label_index_len//2].to(val_data.x.device)


    pos_train_preds = []
    neg_train_preds = []
    pos_test_preds = []
    neg_test_preds = []
    pos_valid_preds = []
    neg_valid_preds = []

    # for perm in tqdm(DataLoader(range(train_source.size(0)), batch_size)):
    #     src, tar = train_source[perm], train_target[perm], 
    #     src_neg, tar_neg = neg_train_edge[perm, 0], neg_train_edge[perm, 1]
    #     tokenized_text_src = get_tokenized_data(src)
    #     tokenized_text_tar = get_tokenized_data(tar)
    #     tokenized_text_src_neg = get_tokenized_data(src_neg)
    #     tokenized_text_tar_neg = get_tokenized_data(tar_neg)

    #     src_pos_emb = model(tokenized_text_src)
    #     tar_pos_emb = model(tokenized_text_tar)
    #     src_neg_emb = model(tokenized_text_src_neg)
    #     tar_neg_emb = model(tokenized_text_tar_neg)

    #     pos_score = predictor(src_pos_emb, tar_pos_emb)
    #     neg_score = predictor(src_neg_emb, tar_neg_emb)

    #     pos_train_preds += [pos_score.squeeze().cpu()]
    #     neg_train_preds += [neg_score.squeeze().cpu()]
    # pos_train_pred = torch.cat(pos_train_preds, dim=0)
    # neg_train_pred = torch.cat(neg_train_preds, dim=0)

    # train_rocauc = roc_auc_score(train_data.edge_label.cpu().tolist(), torch.cat([pos_train_pred, neg_train_pred]).cpu().tolist())

    for perm in tqdm(DataLoader(range(test_source.size(0)), batch_size)):
        src, tar = test_source[perm], test_target[perm], 
        src_neg, tar_neg = neg_test_edge[perm, 0], neg_test_edge[perm, 1]
        tokenized_text_src = get_tokenized_data(src)
        tokenized_text_tar = get_tokenized_data(tar)
        tokenized_text_src_neg = get_tokenized_data(src_neg)
        tokenized_text_tar_neg = get_tokenized_data(tar_neg)

        src_pos_emb = model(tokenized_text_src)
        tar_pos_emb = model(tokenized_text_tar)
        src_neg_emb = model(tokenized_text_src_neg)
        tar_neg_emb = model(tokenized_text_tar_neg)

        pos_score = predictor(src_pos_emb, tar_pos_emb)
        neg_score = predictor(src_neg_emb, tar_neg_emb)

        pos_test_preds += [pos_score.squeeze().cpu()]
        neg_test_preds += [neg_score.squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    # test_rocauc = roc_auc_score(test_data.edge_label.cpu().tolist(), torch.cat([pos_test_pred, neg_test_pred]).cpu().tolist())
    result = {}
    # result['train_rocauc'] = roc_auc_score(train_data.edge_label.cpu().tolist(), torch.cat([pos_train_pred, neg_train_pred]).cpu().tolist())
    # result['valid_rocauc'] = roc_auc_score(val_data.edge_label.cpu().tolist(), torch.cat([pos_valid_pred, neg_valid_pred]).cpu().tolist())
    result['test_rocauc'] = roc_auc_score(test_data.edge_label.cpu().tolist(), torch.cat([pos_test_pred, neg_test_pred]).cpu().tolist())

    # result['train_hitK'] = eval_hits(pos_train_pred, neg_train_pred, K)
    # result['valid_hitK'] = eval_hits(pos_valid_pred, neg_valid_pred, K)
    result['test_hitK'] = eval_hits(pos_test_pred, neg_test_pred, K=100)

    # result['train_mrr'] = eval_hits(pos_train_pred, neg_train_pred, K)
    # result['valid_mrr'] = eval_hits(pos_valid_pred, neg_valid_pred, K)
    # result['test_mrr'] = eval_hits(pos_test_pred, neg_test_pred, K)

    # result['train_mrr'] = eval_mrr(pos_train_pred, neg_train_pred)
    # result['valid_mrr'] = eval_mrr(pos_valid_pred, neg_valid_pred)
    result['test_mrr'] = eval_mrr(pos_test_pred, neg_test_pred)
    # for perm in tqdm(DataLoader(range(valid_source.size(0)), batch_size)):
    #     src, tar = valid_source[perm], valid_target[perm], 
    #     src_neg, tar_neg = neg_valid_edge[perm, 0], neg_valid_edge[perm, 1]
    #     tokenized_text_src = get_tokenized_data(src)
    #     tokenized_text_tar = get_tokenized_data(tar)
    #     tokenized_text_src_neg = get_tokenized_data(src_neg)
    #     tokenized_text_tar_neg = get_tokenized_data(tar_neg)

    #     src_pos_emb = model(tokenized_text_src)
    #     tar_pos_emb = model(tokenized_text_tar)
    #     src_neg_emb = model(tokenized_text_src_neg)
    #     tar_neg_emb = model(tokenized_text_tar_neg)

    #     pos_score = predictor(src_pos_emb, tar_pos_emb)
    #     neg_score = predictor(src_neg_emb, tar_neg_emb)

    #     pos_valid_preds += [pos_score.squeeze().cpu()]
    #     neg_valid_preds += [neg_score.squeeze().cpu()]
    # pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    # neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    # valid_rocauc = roc_auc_score(val_data.edge_label.cpu().tolist(), torch.cat([pos_valid_pred, neg_valid_pred]).cpu().tolist())

    return result
    # return train_rocauc, valid_rocauc, test_rocauc



def train_model():
    # Tokenize the input text
    # input_ids = [tokenizer.encode(text, padding='max_length', max_length=128, return_tensors='pt') for text in X]
    
    # Compute the loss and accuracy during training
    
    # dataset = TensorDataset(torch.cat(input_ids, dim=0), torch.tensor(y))
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # val_input_ids = [tokenizer.encode(text, padding='max_length', max_length=128, return_tensors='pt') for text in X_val]
    # val_dataset = TensorDataset(torch.cat(val_input_ids, dim=0), torch.tensor(y_val))
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    itr = 0
    best_auc = 0
    label_index_len = len(train_data.edge_label_index[0])
    source_edge = train_data.edge_label_index[0,:label_index_len//2].to(train_data.x.device)
    target_edge = train_data.edge_label_index[1,:label_index_len//2].to(train_data.x.device)
    dataloader_train = DataLoader(range(source_edge.size(0)), batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay_factor)
    total_step = len(dataloader_train) * num_epochs
    scheduler = get_warmup_scheduler(optimizer, int(total_step * warm_up_ratio))
    for epoch in range(num_epochs):
        # Training loop
        total_loss = 0
        total_examples = 0
        for perm in tqdm(dataloader_train):
            itr += 1
            model.train()
            optimizer.zero_grad()
            src, tar = source_edge[perm], target_edge[perm]
            tar_neg = torch.randint(0, train_data.num_nodes, src.size(), dtype=torch.long, device=train_data.x.device)
            tokenized_text_src = get_tokenized_data(src)
            tokenized_text_tar_pos = get_tokenized_data(tar)
            tokenized_text_tar_neg = get_tokenized_data(tar_neg)
            
            src_emb = model(tokenized_text_src)
            tar_pos_emb = model(tokenized_text_tar_pos)
            pos_score = predictor(src_emb, tar_pos_emb)
            pos_loss = -torch.log(pos_score + 1e-15).mean()

            tar_neg_emb = model(tokenized_text_tar_neg)
            neg_score = predictor(src_emb, tar_neg_emb)
            neg_loss = -torch.log(1 - neg_score + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            num_examples = pos_score.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
            logging.info(f'')
            logging.info(f'iteration {itr+1}: train loss: {loss.item()}, lr: {scheduler.get_last_lr()[0]}')
            # print(f'pos accuracy:{((pos_score>0.5).sum() / len(pos_score)).item()}, neg accuracy:{((neg_score<=0.5).sum() / len(pos_score)).item()}, loss: {loss.item()}')
            print(f'pos score:{(pos_score).mean().item()}, neg score:{(neg_score).mean().item()}, loss: {loss.item()}')

            if not itr % eval_every:
                # Validation loop
                # train_roc_auc, valid_roc_auc, test_roc_auc = evaluation()
                # logging.info(f'iteration: {itr+1}, Loss: {loss:.4f}, Train: {100 * train_roc_auc:.4f}%, Valid: {100 * valid_roc_auc:.4f}%, Test: {100 * test_roc_auc:.4f}%')
                result = evaluation()
                auc, hit, mrr = result['test_rocauc'], result['test_hitK'], result['test_mrr']
                logging.info(f'iteration: {itr+1}, Loss: {loss:.4f}, Test AUC: {100 * auc:.4f}, Hits 100:  {100 * hit:.4f}, Mrr: {100 * mrr:.4f}')

                if best_auc < result['test_rocauc']:
                    best_auc = result['test_rocauc']
                    if not os.path.exists('save_model'):
                        os.makedirs('save_model', exist_ok=True)
                    # torch.save(model, f'save_model/t5-base-link-prediction')
                    torch.save(model, args.save_path)


# Train the model
train_model()

