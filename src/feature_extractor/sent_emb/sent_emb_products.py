from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import os
ds = load_dataset("Egbertjing/products_full")
text = ds['train']['text']

emb_model = SentenceTransformer('sentence-t5-base', device='cuda:0')
feature_emb_title_abs = emb_model.encode(text, show_progress_bar=True)
emb = torch.tensor(feature_emb_title_abs).to('cpu')
torch.save(emb, '../../../emb/sent_emb_products.pt')
