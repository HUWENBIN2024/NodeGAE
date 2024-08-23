from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import os
ds = load_dataset("hubin/arxiv_title_abstract_all_for_train")
text = ds['train']['text']

emb_model = SentenceTransformer('sentence-t5-base', device='cuda:0')
feature_emb_title_abs = emb_model.encode(text, show_progress_bar=True)
os.makedirs('emb', exist_ok=True)
torch.save(feature_emb_title_abs, 'emb/sent_emb.pt')
