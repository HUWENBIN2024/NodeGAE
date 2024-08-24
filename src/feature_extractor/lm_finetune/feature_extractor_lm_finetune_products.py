import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from simteg.src.dataset import load_dataset
import numpy as np
import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
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

parser = argparse.ArgumentParser(description='feature extractor')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_path', type=str, default='save_lm_finetune/t5base_mode_cls.pt')
parser.add_argument('--save_path', type=str, default='../../../emb/lm_finetune_products.pt')

args = parser.parse_args()

device = args.device


print('Loading the dataset...')
dataset_name = 'products'
dataset = load_dataset(name='ogbn-products')
data = dataset._data

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
        self.t5_model = t5_model
        # Define layers here
        self.mlp = nn.Sequential( 
            nn.Linear(in_features=768, out_features=256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=40),
        )   
    def forward(self, x):
        outputs = self.t5_model(**x)
        embeddings = average_pool(outputs.last_hidden_state, x['attention_mask'])
        logits = self.mlp(embeddings)
        return logits


# # Tokenize the input texts

# model  = e5_cls_model()
try:
    model  = e5_cls_model()
    model.load_state_dict(torch.load(args.model_path))
except:
    model = torch.load(args.model_path)

model.to(device)


def get_emb_from_t5_model():
    text_list = df_id2text['text'].to_list()
    emb_list = []
    print('encoding the sentences using finetuned t5 base ...')
    with torch.no_grad():
        for i in tqdm(range(len(text_list))):
            text = text_list[i]
            tokenized_text = tokenizer(text, padding='max_length', max_length=512, return_tensors='pt', truncation=True).to(device)
            outputs = model.t5_model(**tokenized_text)
            embeddings = average_pool(outputs.last_hidden_state, tokenized_text['attention_mask']).reshape(-1) # shape: (1024, )
            emb_list.append(embeddings)
        emb = torch.stack(emb_list, dim=0) # shape (# data, 1024)
    return emb


emb = get_emb_from_t5_model()
torch.save(emb, args.save_path)
