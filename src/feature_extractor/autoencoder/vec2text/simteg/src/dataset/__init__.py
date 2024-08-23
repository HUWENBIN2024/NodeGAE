from ogb.nodeproppred import Evaluator as NodeEvaluator
import os
from .ogbn_products import OgbnProductsWithText

_root = 'dataset'
def load_dataset(name, root=_root, tokenizer=None, tokenize=True):
    return OgbnProductsWithText(root=root, tokenizer=tokenizer, tokenize=tokenize)

