import sys
sys.path.append('src/dataset/products/')
print(sys.path)
from simteg.src.dataset import load_dataset

dataset = load_dataset(name='ogbn-products')
print(dataset._data.x.shape)