import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import logging
import os


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_size_gnn, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_size_gnn, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_size_gnn))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_size_gnn, hidden_size_gnn, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_size_gnn))
        self.convs.append(GCNConv(hidden_size_gnn, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_size_gnn, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_size_gnn))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_size_gnn))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size_gnn, hidden_size_gnn))
            self.bns.append(torch.nn.BatchNorm1d(hidden_size_gnn))
        self.convs.append(SAGEConv(hidden_size_gnn, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer, label_smoothing=0):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.cross_entropy(out, data.y.squeeze(1)[train_idx], label_smoothing=label_smoothing)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():

    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    # parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_size_gnn', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='sage')

    parser.add_argument('--label_smoothing', type=float, default=0) # 0.3
    parser.add_argument('--weight_decay', type=float, default=0) # 4e-6
    
    parser.add_argument('--is_emb_from_path', type=bool, default=True)
    parser.add_argument('--emb_path', type=str, default='emb/nodegae_feature_emb.pt')
    parser.add_argument('--log_path', type=str, default='logs/gnn_node_classification.logs')

    args = parser.parse_args()
    print(args)

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename=args.log_path, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


    use_emb_from_path = args.is_emb_from_path
    emb_path = args.emb_path
    label_smoothing = args.label_smoothing
    weight_decay = args.weight_decay
    print(f'lr:{args.lr}, dropout:{args.dropout}, num_layers:{args.num_layers}, hidden: {args.hidden_size_gnn}, label smooth: {label_smoothing}, weight_decay={weight_decay}')
    logging.info(f'lr:{args.lr}, dropout:{args.dropout}, num_layers:{args.num_layers}, hidden: {args.hidden_size_gnn}, label smooth: {label_smoothing}, weight_decay={weight_decay}')

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]), root='datasets')

    data = dataset[0]
    # data.adj_t = data.adj_t.to_symmetric()
    if use_emb_from_path:
        data.x = torch.load(emb_path, map_location=device)
    data = data.to(device)

    split_idx = dataset.get_idx_split() # train:test:valid = 90941/48603/29799
    train_idx = split_idx['train'].to(device)

    if args.model_name == 'sage':
        model = SAGE(data.num_features, args.hidden_size_gnn,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    elif args.model_name =='gcn':
        model = GCN(data.num_features, args.hidden_size_gnn,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    best_acc = 0
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        for epoch in range(1, 1 + args.num_epochs):
            loss = train(model, data, train_idx, optimizer, label_smoothing)
            result = test(model, data, split_idx, evaluator)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                logging.info(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%, Test: {100 * test_acc:.2f}%')
                if best_acc < test_acc:
                    best_acc = test_acc
    print('best result:', best_acc)
    logging.info(f'best result: {best_acc}\n')
        # torch.save(model.state_dict(), 'arxiv.pt')



if __name__ == "__main__":
    main()
