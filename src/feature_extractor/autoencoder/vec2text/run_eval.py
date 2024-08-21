import transformers
from experiments import experiment_from_args
from run_args import DataArguments, ModelArguments, TrainingArguments
import torch
import torch_geometric.transforms as T
import sys  
import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator




def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run_eval()


if __name__ == "__main__":
    main()
