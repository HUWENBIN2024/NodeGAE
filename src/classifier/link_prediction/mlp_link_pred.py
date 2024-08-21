import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from sklearn.metrics import roc_auc_score
import argparse
import os
import sys
# sys.path.append('.')
from utils import eval_hits, eval_mrr

parser = argparse.ArgumentParser(description='mlp')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--is_emb_from_path', type=bool, default=True)
parser.add_argument('--log_path', type=str, default='logs/mlp_link_prediction.logs')
parser.add_argument('--emb_path', type=str, default='emb/nodegae_feature_emb.pt')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--hidden_size_gnn', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=1024)

args = parser.parse_args()

device = args.device
lr = args.lr
num_epochs = args.num_epoch
bs = args.batch_size
log_file_path = args.log_path
emb_from_pt = args.is_emb_from_path

emb_path = args.emb_path
hidden_size = args.hidden_size_gnn

dataset=PygNodePropPredDataset(name='ogbn-arxiv', transform=T.Compose([T.ToUndirected()]), root='datasets')
data = dataset[0]

data_path = 'datasets/ogbn_arxiv'
train_data = torch.load(os.path.join(data_path,'split/link_pred/train')).to(device)
test_data = torch.load(os.path.join(data_path,'split/link_pred/test')).to(device)
val_data = torch.load(os.path.join(data_path,'split/link_pred/val')).to(device)

counter = 0

if emb_from_pt:
    print('load the emb from the path')
    fea_emb_all_data = torch.load(emb_path).to(device)
    data.x = fea_emb_all_data
    train_data.x = fea_emb_all_data
    test_data.x = fea_emb_all_data
    val_data.x = fea_emb_all_data

def update_data(data):
    '''
    update the data for training
    '''
    train_data.x = data.x
    test_data.x = data.x
    val_data.x = data.x

    

def train(predictor, optimizer, batch_size, evaluator = None):
    # model.train()
    predictor.train()

    # source_edge = split_edge['train']['source_node'].to(data.x.device)
    # target_edge = split_edge['train']['target_node'].to(data.x.device)
    label_index_len = len(train_data.edge_label_index[0])
    source_edge = train_data.edge_label_index[0,:label_index_len//2].to(train_data.x.device)
    target_edge = train_data.edge_label_index[1,:label_index_len//2].to(train_data.x.device)

    # print(source_edge.shape, target_edge.shape)
    # print(train_data.edge_label[:label_index_len//2])
    # print(train_data.edge_label[label_index_len//2:])
    # exit(-1)
    global counter
    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(source_edge.size(0)), batch_size, shuffle=True)):
        optimizer.zero_grad()

        # h = model(train_data.x, train_data.edge_index)

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(train_data.x[src], train_data.x[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, train_data.num_nodes, src.size(),
                                dtype=torch.long, device=train_data.x.device)
        neg_out = predictor(train_data.x[src], train_data.x[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        
        if not counter % 2:
            result = test(predictor, evaluator, bs)
            train_roc_auc, valid_roc_auc, test_roc_auc = result['train_rocauc'], result['valid_rocauc'], result['test_rocauc']
            hit_k_score = result['test_hitK']
            mrr_score = result['test_mrr']
            # hit_k_score = test_hit_k(model, k=50)
            print(f'itr: {counter:02d}, Loss: {loss:.4f}, Train: {100 * train_roc_auc:.4f}%, Valid: {100 * valid_roc_auc:.4f}%, Test: {100 * test_roc_auc:.4f}%, Hit@100: {100 * hit_k_score:.4f}%, MRR: {100 * mrr_score:.4f}%')
            logging.info(f'itr: {counter:02d}, Loss: {loss:.4f}, Train: {100 * train_roc_auc:.4f}%, Valid: {100 * valid_roc_auc:.4f}%, Test: {100 * test_roc_auc:.4f}%, Hit@100: {100 * hit_k_score:.4f}%, MRR: {100 * mrr_score:.4f}%')
        counter += 1
        # print(f'Loss: {loss.item()}')
        # print(f'pos accuracy:{((pos_out>0.5).sum() / len(pos_out)).item()}')
        # print(f'neg accuracy:{((neg_out<=0.5).sum() / len(pos_out)).item()}')


    return total_loss / total_examples


# @torch.no_grad()
# def test(model, predictor, evaluator, batch_size):
#     predictor.eval()

#     h = model(test_data.x, test_data.edge_index)

#     def test_split(split):
#         if split == 'eval_train':
#             label_index_len = len(train_data.edge_label_index[0])
#             source = train_data.edge_label_index[0,:label_index_len//2].to(train_data.x.device)
#             target = train_data.edge_label_index[1,:label_index_len//2].to(train_data.x.device)
#             source_neg = train_data.edge_label_index[0,label_index_len//2:].to(train_data.x.device)
#             target_neg = train_data.edge_label_index[0,label_index_len//2:].to(train_data.x.device)
#         elif split == 'valid':
#             label_index_len = len(val_data.edge_label_index[0])
#             source = val_data.edge_label_index[0,:label_index_len//2].to(val_data.x.device)
#             target = val_data.edge_label_index[1,:label_index_len//2].to(val_data.x.device)
#             source_neg = val_data.edge_label_index[0,label_index_len//2:].to(val_data.x.device)
#             target_neg = val_data.edge_label_index[0,label_index_len//2:].to(val_data.x.device)
#         elif split == 'test':
#             label_index_len = len(test_data.edge_label_index[0])
#             source = test_data.edge_label_index[0,:label_index_len//2].to(test_data.x.device)
#             target = test_data.edge_label_index[1,:label_index_len//2].to(test_data.x.device)
#             source_neg = test_data.edge_label_index[0,label_index_len//2:].to(test_data.x.device)
#             target_neg = test_data.edge_label_index[0,label_index_len//2:].to(test_data.x.device)


#         pos_preds = []
#         for perm in DataLoader(range(source.size(0)), batch_size):
#             src, dst = source[perm], target[perm]
#             pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
#         pos_pred = torch.cat(pos_preds, dim=0)

#         neg_preds = []

#         source_neg = source_neg.view(-1, 1).repeat(1, 1000).view(-1)
#         target_neg = target_neg.view(-1)
#         for perm_ in tqdm(DataLoader(range(source_neg.size(0)), batch_size)):
#             src_neg, dst_neg = source_neg[perm_], target_neg[perm_]
#             predictor(h[src_neg], h[dst_neg])
#             neg_preds += [predictor(h[src_neg], h[dst_neg]).squeeze().cpu()]
#         neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

#         return evaluator.eval({
#             'y_pred_pos': pos_pred,
#             'y_pred_neg': neg_pred,
#         })['mrr_list'].mean().item()

#     train_mrr = test_split('eval_train')
#     valid_mrr = test_split('valid')
#     test_mrr = test_split('test')

#     return train_mrr, valid_mrr, test_mrr
train_label_index_len = len(train_data.edge_label_index[0])
valid_label_index_len = len(val_data.edge_label_index[0])
test_label_index_len = len(test_data.edge_label_index[0])

neg_train_edge = torch.randint(0, train_data.num_nodes, (train_label_index_len//2, 2), dtype=torch.long, device=train_data.x.device)
neg_valid_edge = torch.randint(0, val_data.num_nodes, (valid_label_index_len//2, 2), dtype=torch.long, device=train_data.x.device)
neg_test_edge = torch.randint(0, test_data.num_nodes, (test_label_index_len//2, 2), dtype=torch.long, device=train_data.x.device)


@torch.no_grad()
def test(predictor, evaluator, batch_size, K=50):
    # model.eval()
    predictor.eval()

    # h = model(test_data.x, test_data.edge_index)

    train_label_index_len = len(train_data.edge_label_index[0])
    train_source = train_data.edge_label_index[0,:train_label_index_len//2].to(train_data.x.device)
    train_target = train_data.edge_label_index[1,:train_label_index_len//2].to(train_data.x.device)
    pos_train_edge = torch.stack([train_source, train_target], dim=1)
    # train_source_neg = train_data.edge_label_index[0,train_label_index_len//2:].to(train_data.x.device)
    # train_target_neg = train_data.edge_label_index[0,train_label_index_len//2:].to(train_data.x.device)
    # neg_train_edge = torch.stack([train_source_neg, train_target_neg], dim=1)

    valid_label_index_len = len(val_data.edge_label_index[0])
    valid_source = val_data.edge_label_index[0,:valid_label_index_len//2].to(val_data.x.device)
    valid_target = val_data.edge_label_index[1,:valid_label_index_len//2].to(val_data.x.device)
    pos_valid_edge = torch.stack([valid_source, valid_target], dim=1)
    # valid_source_neg = val_data.edge_label_index[0,valid_label_index_len//2:].to(val_data.x.device)
    # valid_target_neg = val_data.edge_label_index[0,valid_label_index_len//2:].to(val_data.x.device)
    # neg_valid_edge = torch.stack([valid_source_neg, valid_target_neg], dim=1)


    test_label_index_len = len(test_data.edge_label_index[0])
    test_source = test_data.edge_label_index[0,:test_label_index_len//2].to(test_data.x.device)
    test_target = test_data.edge_label_index[1,:test_label_index_len//2].to(test_data.x.device)
    pos_test_edge = torch.stack([test_source, test_target], dim=1)
    # test_source_neg = test_data.edge_label_index[0,test_label_index_len//2:].to(test_data.x.device)
    # test_target_neg = test_data.edge_label_index[0,test_label_index_len//2:].to(test_data.x.device)
    # neg_test_edge = torch.stack([test_source_neg, test_target_neg], dim=1)



    # pos_train_edge = split_edge['train']['edge'].to(h.device)
    # neg_train_edge = split_edge['train']['edge_neg'].to(h.device)
    # pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    # neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    # pos_test_edge = split_edge['test']['edge'].to(h.device)
    # neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t() # [2, 1024]
        pos_train_preds += [predictor(test_data.x[edge[0]], test_data.x[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    # logging.info(f'pos train: {pos_train_pred.mean().item()}')


    neg_train_preds = []
    for perm in DataLoader(range(neg_train_edge.size(0)), batch_size):
        edge = neg_train_edge[perm].t()
        neg_train_preds += [predictor(test_data.x[edge[0]], test_data.x[edge[1]]).squeeze().cpu()]
    neg_train_pred = torch.cat(neg_train_preds, dim=0)
    # logging.info(f'neg train: {neg_train_pred.mean().item()}')

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(test_data.x[edge[0]], test_data.x[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    # logging.info(f'pos val: {pos_valid_pred.mean().item()}')


    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(test_data.x[edge[0]], test_data.x[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)
    # logging.info(f'neg val: {neg_valid_pred.mean().item()}')


    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(test_data.x[edge[0]], test_data.x[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    # logging.info(f'pos test: {pos_test_pred.mean().item()}')


    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(test_data.x[edge[0]], test_data.x[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    # logging.info(f'neg test: {neg_test_pred.mean().item()}')

    result = {}
    result['train_rocauc'] = roc_auc_score(train_data.edge_label.cpu().tolist(), torch.cat([pos_train_pred, neg_train_pred]).cpu().tolist())
    result['valid_rocauc'] = roc_auc_score(val_data.edge_label.cpu().tolist(), torch.cat([pos_valid_pred, neg_valid_pred]).cpu().tolist())
    result['test_rocauc'] = roc_auc_score(test_data.edge_label.cpu().tolist(), torch.cat([pos_test_pred, neg_test_pred]).cpu().tolist())

    result['train_hitK'] = eval_hits(pos_train_pred, neg_train_pred, K)
    result['valid_hitK'] = eval_hits(pos_valid_pred, neg_valid_pred, K)
    result['test_hitK'] = eval_hits(pos_test_pred, neg_test_pred, K)

    result['train_mrr'] = eval_hits(pos_train_pred, neg_train_pred, K)
    result['valid_mrr'] = eval_hits(pos_valid_pred, neg_valid_pred, K)
    result['test_mrr'] = eval_hits(pos_test_pred, neg_test_pred, K)

    result['train_mrr'] = eval_mrr(pos_train_pred, neg_train_pred)
    result['valid_mrr'] = eval_mrr(pos_valid_pred, neg_valid_pred)
    result['test_mrr'] = eval_mrr(pos_test_pred, neg_test_pred)
    
    return result



# class SAGE_LINK_PRED(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
#                  dropout):
#         super(SAGE_LINK_PRED, self).__init__()

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(SAGEConv(in_channels, hidden_channels))
#         for _ in range(num_layers - 2):
#             self.convs.append(SAGEConv(hidden_channels, hidden_channels))
#         self.convs.append(SAGEConv(hidden_channels, out_channels))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()

#     def forward(self, x, adj_t):
#         for conv in self.convs[:-1]:
#             x = conv(x, adj_t)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
#         return x
    
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
    
def main():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'lr: {lr}, batch size: {bs}, hidden size: {hidden_size}')

    # model = SAGE_LINK_PRED(data.num_features,256, 256, 3, 0).to(device)
    predictor = LinkPredictor(data.num_features, hidden_size, 1, 3, 0).to(device)

    evaluator = Evaluator(name='ogbl-vessel')

    # eval_steps = 1
    # model.reset_parameters()
    predictor.reset_parameters()
    optimizer = torch.optim.Adam(list(predictor.parameters()), lr=lr)
        # list(model.parameters()) + list(predictor.parameters()), lr=lr)

    for epoch in range(num_epochs):
        loss = train(predictor,  optimizer, bs, evaluator)
        result = test(predictor, evaluator, bs)
        train_roc_auc, valid_roc_auc, test_roc_auc = result['train_rocauc'], result['valid_rocauc'], result['test_rocauc']
        hit_k_score = result['test_hitK']
        mrr_score = result['test_mrr']
        # hit_k_score = test_hit_k(model, k=50)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_roc_auc:.4f}%, Valid: {100 * valid_roc_auc:.4f}%, Test: {100 * test_roc_auc:.4f}%, Hit@100: {100 * hit_k_score:.4f}%, MRR: {100 * mrr_score:.4f}%')
        logging.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_roc_auc:.4f}%, Valid: {100 * valid_roc_auc:.4f}%, Test: {100 * test_roc_auc:.4f}%, Hit@100: {100 * hit_k_score:.4f}%, MRR: {100 * mrr_score:.4f}%')
        # torch.save(model.state_dict(), 'eval_result/model_weights.pth')
        # torch.save(predictor.state_dict(), 'eval_result/predictor_weights.pth')
        

if __name__ == "__main__":
    main()