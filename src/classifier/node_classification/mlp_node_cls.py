import sys
from torch import nn


from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import torch

import logging
import argparse
import os

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_gnn, output_size, num_layers):
        super(MLP, self).__init__()

        self.num_layers = num_layers
        
        self.layers = torch.nn.ModuleList()
        self.activate = torch.nn.ModuleList()
        # Define the layers
        self.layers.append(nn.Linear(input_size, hidden_size_gnn))
        self.activate.append(nn.ReLU())
        for i in range(1, num_layers-1):
            self.layers.append(nn.Linear(hidden_size_gnn, hidden_size_gnn))
            self.activate.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_size_gnn, output_size))
        
    def forward(self, x):
        # Forward pass
        for i in range(self.num_layers-1):
            x = self.layers[i](x)
            x = self.activate[i](x)
        x = self.layers[self.num_layers-1](x)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def train(model, data_x, target_train, optimizer, criterion, scheduler=None):
    model.train()

    optimizer.zero_grad()
    out = model(data_x)
    loss = criterion(out, target_train)
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    return loss.item()


@torch.no_grad()
def test(model, data_x, target_test):
    model.eval()

    out = model(data_x)
    y_pred = out.argmax(dim=-1, keepdim=True).reshape(-1,)
    acc = ((y_pred == target_test).sum().item()) / len(data_x)

    return acc


def main():
    parser = argparse.ArgumentParser(description='mlp')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--is_emb_from_path', type=bool, default=True)
    parser.add_argument('--log_path', type=str, default='logs/mlp_node_classification.logs')
    parser.add_argument('--emb_path', type=str, default='emb/nodegae_feature_emb.pt')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--hidden_size_gnn', type=int, default=256)
    args = parser.parse_args()

    device = args.device
    emb_from_pt = args.is_emb_from_path
    emb_path = args.emb_path
    log_file_path = args.log_path
    lr = args.lr
    num_epoch = args.num_epoch
    hidden_size_gnn = args.hidden_size_gnn

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'lr: {lr}, hidden size: {hidden_size_gnn}')

    dataset=PygNodePropPredDataset(name='ogbn-arxiv', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]), root='datasets')
    data = dataset[0]

    split_idx = dataset.get_idx_split() # train:test:valid = 90941/48603/29799
    train_split = split_idx['train'].to(device)
    test_split = split_idx['test'].to(device)

    if emb_from_pt:
        print('load the emb from the path')
        fea_emb_all_data = torch.load(emb_path).to(device)
        data.x = fea_emb_all_data

    model = MLP(data.x.size(1), hidden_size_gnn, dataset.num_classes, num_layers=2).to(device).to(torch.bfloat16)
    model.apply(init_weights)

    data = data.to(device)
    data.x = data.x.to(torch.bfloat16)
    data.adj_t = data.adj_t.to(torch.bfloat16)


    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(name='ogbn-arxiv')

    # with torch.no_grad():
    #     graph_emb_train = graph_model(data.x, data.adj_t)[train_split]
    #     graph_emb_test = graph_model(data.x, data.adj_t)[test_split]
    emb_train = data.x[train_split]
    emb_test = data.x[test_split]

    target_train = data.y[train_split].reshape(-1,)
    target_test = data.y[test_split].reshape(-1,)

    # num_warmup_steps = num_epoch // 2
    num_training_steps = num_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    # scheduler = get_warmup_scheduler(optimizer, num_warmup_steps)

    best_acc = 0
    for epoch in range(1, num_epoch+1):
        
        loss = train(model, emb_train, target_train,  optimizer, criterion)
        acc = test(model, emb_test, target_test)

        print('ep', epoch, 'loss:', loss, 'acc:', acc)
        logging.info(f'ep: {epoch}, loss: {loss}, acc: {acc}')
        if best_acc < acc:
            best_acc = acc

    print(f'best acc is: {best_acc}')
    logging.info(f'best acc is: {best_acc}')


if __name__ == "__main__":
    main()
