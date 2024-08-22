import sys
sys.path.append("../")
# import vec2text
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR
from torch_geometric.loader import NeighborLoader

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import torch
import torch.nn.functional as F
import vec2text.models.inversion as inversion
from torch.utils.data import Dataset, DataLoader

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from simteg.src.dataset import load_dataset

from torch_geometric.utils import to_undirected
import random
import numpy as np
import gzip
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from analyze_utils import load_experiment_and_trainer
from transformers import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch import Tensor
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='feature extractor')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='saves/autoencoder/checkpoint-200000')
    parser.add_argument('--save_path', type=str, default='../../../../emb/nodegae_feature_emb.pt')

    args = parser.parse_args()

    os.makedirs('../../../../emb', exist_ok=True)

    device = args.device
    model_path = args.model_path
    max_seq_len = 256

    dataset=PygNodePropPredDataset(name='ogbn-products', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]), root='../../../../datasets')
    data = dataset[0]
    data = data.to(device)

    split_idx = dataset.get_idx_split() # train:test:valid = 90941/48603/29799
    train_split = split_idx['train'].to(device)
    test_split = split_idx['test'].to(device)

    print('loading the graph meta data...')
    dataset = load_dataset(name='ogbn-products', tokenizer='sentence-transformers/sentence-t5-base')
    data = dataset._data

    print('loading the node meta data...')
    input_ids = data.input_ids
    attention_mask = data.attention_mask
    df = pd.read_csv('/home/hjingaa/dataset/ogbn_products_text/raw/Amazon-3M.raw/products.csv', sep=" ")
    df.replace(np.nan, "", inplace=True)
    df["titlecontent"] = df["title"] + ". " + df["content"]
    df = df.drop(columns=["title", "content"])
    df.rename(columns={"uid": "asin"}, inplace=True)
    df_mapping = pd.read_csv("/home/hjingaa/dataset/ogbn_products_text/mapping/nodeidx2asin.csv.gz")
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

    class encoder(nn.Module):
        def __init__(self,):
            experiment, trainer = load_experiment_and_trainer(model_path)
            super(encoder, self).__init__()
            self.embedder_encoder = trainer.model.embedder_encoder
            self.tokenizer_t5 = trainer.model.tokenizer_t5
            
        def forward(self, node_idx):
            title_list = []
            for idx_ in node_idx:
                idx_int = idx_.item()
                
                title_list.append(df_id2text.iloc[idx_int]['text'])
            input_ids_title_list = self.tokenizer_t5(title_list, padding=True, truncation=True, return_tensors="pt" ,max_length=max_seq_len).to(device)
            center_sent_embs_batch_last_hidden_state = self.embedder_encoder(input_ids=input_ids_title_list['input_ids'], attention_mask=input_ids_title_list['attention_mask']).last_hidden_state
            center_sent_embs_batch = average_pool(center_sent_embs_batch_last_hidden_state, input_ids_title_list['attention_mask'])
            
            return center_sent_embs_batch

    fea_emb_list = []
    model = encoder()
    model = model.to(device)
    print('preparing the feature embeddings...')
    with torch.no_grad():
        for idx_ in tqdm(range(len(data.x))):
            graph_emb = model([torch.tensor(idx_)]) # shape: (1, 1024)
            fea_emb_list.append(graph_emb.reshape(-1,))
    fea_emb_all_data = torch.stack(fea_emb_list, dim=0) 
    torch.save(fea_emb_all_data, args.save_path)


if __name__ == "__main__":
    main()

