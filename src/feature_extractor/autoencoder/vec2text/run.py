import transformers
from experiments import experiment_from_args
from run_args import DataArguments, ModelArguments, TrainingArguments
import torch
import torch_geometric.transforms as T
import sys  
import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator



# sys.path.append('../')
# from gnn import SAGE 
# class SAGE_embedding(SAGE):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
#         super(SAGE_embedding, self).__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
#     def forward(self, x, adj_t):
#         x = self.convs[0](x, adj_t)
#         x = self.bns[0](x)
#         return x
    
# graph_model = SAGE_embedding(768, 768, 40, 3, 0.5)

# graph_model.load_state_dict(torch.load('/data/hjingaa/github/geia_graph/exchange_files/sage_model_arxiv_768word_emb.pt'))   





def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()


if __name__ == "__main__":
    main()
