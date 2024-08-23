
from ogb.nodeproppred import Evaluator as NodeEvaluator


from .ogbn_products import OgbnProductsWithText
_root = "src/feature_extractor/autoencoder/vec2text/dataset"

def load_dataset(name, root=_root, tokenizer=None, tokenize=True):
    return OgbnProductsWithText(root=root, tokenizer=tokenizer, tokenize=tokenize)

