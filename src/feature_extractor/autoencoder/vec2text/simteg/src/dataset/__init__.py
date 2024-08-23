# from ogb.linkproppred import Evaluator as LinkEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator

# from .ogbl_citation2 import OgblCitation2WithText
# from .ogbn_arxiv import OgbnArxivWithText
# from .ogbn_arxiv_tape import OgbnArxivWithTAPE
from .ogbn_products import OgbnProductsWithText
_root = "/home/hjingaa/github/NodeGAE/src/feature_extractor/autoencoder/vec2text/dataset"

def load_dataset(name, root=_root, tokenizer=None, tokenize=True):
    return OgbnProductsWithText(root=root, tokenizer=tokenizer, tokenize=tokenize)

