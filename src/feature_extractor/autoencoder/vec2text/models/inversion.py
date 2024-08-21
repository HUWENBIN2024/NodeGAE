

import copy
import logging
from typing import Dict, Optional, Tuple
import os

import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from vec2text.models.config import InversionConfig
from vec2text.models.model_utils import (
    FREEZE_STRATEGIES,
    disable_dropout,
    freeze_params,
    load_embedder_and_tokenizer,
    load_encoder_decoder,
    load_tokenizer,
    mean_pool,
)
import os
from vec2text.utils import embed_api
import pandas as pd
import gzip
import random
import numpy as np

# import gnn
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit

num_sampled_neighbour = 5
encoder_input_token_max_len = 256


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# graph_model = gnn.SAGE
# class SAGE_embedding(gnn.SAGE):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_sage_layers: int):
#         super(SAGE_embedding, self).__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)
#         self.num_sage_layers = num_sage_layers
#     def forward(self, x, adj_t):
#         # x = self.convs[0](x, adj_t)
#         # x = self.bns[0](x)
#         # super().forward(x, adj_t)
#         if self.num_sage_layers == 1:
#             x = self.convs[0](x, adj_t)
#             x = self.bns[0](x)
#         elif self.num_sage_layers == 2:
#             x = self.convs[0](x, adj_t)
#             x = self.bns[0](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             x = self.convs[1](x, adj_t)
#             x = self.bns[1](x)
#         elif self.num_sage_layers == 3:
#             x = super().forward(x, adj_t)
#         return x

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
data_path = '../../../../datasets/ogbn_arxiv'
print('loading the graph meta data...')
df_node2paper = pd.read_csv(os.path.join(data_path, 'mapping/nodeidx2paperid.csv.gz'), compression='gzip')
file_path = os.path.join(data_path, 'titleabs.tsv.gz')
df_paperid2titleabs = pd.read_csv(gzip.open(file_path, 'rt'), delimiter='\t', names=['paperid', 'title', 'abstract'])
df_paperid2titleabs = df_paperid2titleabs.drop(0)
df_paperid2titleabs = df_paperid2titleabs.drop(179719)
def fn(x):
    return int(x['paperid'])
df_paperid2titleabs['paperid_int'] = df_paperid2titleabs.apply(fn, axis=1)


    
logger = logging.getLogger(__name__)

# TODO: can we make this class a HF pretrained model so it works nicely with
# .push_to_hub(), etc.?
# TODO: Need config to subclass transformers.PreTrainedModel.
class InversionModel(transformers.PreTrainedModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """

    config_class = InversionConfig
    embedder: nn.Module
    embedder_tokenizer: transformers.PreTrainedTokenizer  # embedder's tokenizer
    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    encoder_decoder_lora: bool  # Whether to use LoRA for the encoder-decoder model
    tokenizer: transformers.PreTrainedTokenizer  # encoder_decoder's tokenizer
    embedding_transform: nn.Module  # Module that transformers embedder output into encoder-decoder input
    bottleneck_dim: int  # Bottleneck dimension for embedding_transform
    num_repeat_tokens: int  # Sequence length for repeating embedder embedding for encoder-decoder input
    embedder_dim: int  # Hidden dimension of embedding model
    embedder_no_grad: bool  # Disable gradients for embedding model
    embedder_fake_with_zeros: bool  # Whether to just provide zeros as input for encoder-decoder (unconditional)
    embedding_transform_strategy: str  # Way to transform bottleneck embedding into input for encoder-decoder
    use_frozen_embeddings_as_input: bool  # Whether to train/evaluate on frozen embeddings
    embedded_tokens: torch.Tensor  # used for decoding
    embedder_model_api: Optional[str]
    current_step = 0

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)

        # config.is_link_prediction = False
        # config.non_blocking = None
        if not config.is_link_prediction:
            print('doing node classification')
            dataset=PygNodePropPredDataset(name='ogbn-arxiv', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]), root='../../../../datasets')
            data = dataset[0]
            print('building adjacency list...')
            adj_list = {}
            sparse_tensor = data.adj_t
            for row in range(sparse_tensor.size(0)):
                neighbors = []
                start = sparse_tensor.crow_indices()[row]
                end = sparse_tensor.crow_indices()[row + 1]
                for idx in range(start, end):
                    neighbors.append(sparse_tensor.col_indices()[idx].item())
                adj_list[row] = neighbors
        else:
            print('doing link prediction')
            dataset=PygNodePropPredDataset(name='ogbn-arxiv', transform=T.Compose([T.ToUndirected()]), root='../../../../datasets')
            data = dataset[0]
            if not os.path.exists(os.path.join(data_path, 'split/link_pred')):
                os.makedirs(os.path.join(data_path, 'split/link_pred'))
                tfs = RandomLinkSplit(is_undirected=True, 
                        add_negative_train_samples=True,
                          neg_sampling_ratio=1.0,
                        key = "edge_label", # supervision label
                        disjoint_train_ratio=0,# disjoint mode if > 0
                        # edge_types=None, # for heteroData
                        # rev_edge_types=None, # for heteroData
                        num_val = 0.2,
                        num_test = 0.1,
                        )
                train_data, val_data, test_data = tfs(data)
                torch.save(train_data, os.path.join(data_path, 'split/link_pred/train'))
                torch.save(test_data, os.path.join(data_path, 'split/link_pred/test'))
                torch.save(val_data, os.path.join(data_path, 'split/link_pred/val'))
            else:
                train_data = torch.load(os.path.join(data_path, 'split/link_pred/train'))
        
        
            print('building adjacency list...')
            adj_list = {}
            for i in range(len(train_data.x)):
                adj_list[i] = [i]
            for i in range(train_data.edge_index.shape[1]):
                src, tar = train_data.edge_index[:,i].tolist()
                adj_list[src].append(tar)

        self.adj_list = adj_list

        embedder_model_api = config.embedder_model_api
        embedder_fake_with_zeros = config.embedder_fake_with_zeros
        use_frozen_embeddings_as_input = config.use_frozen_embeddings_as_input
        encoder_dropout_disabled = config.encoder_dropout_disabled
        decoder_dropout_disabled = config.decoder_dropout_disabled
        embeddings_from_layer_n = config.embeddings_from_layer_n

        encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
            lora=config.use_lora,
        )

        # embedder, embedder_tokenizer = load_embedder_and_tokenizer(
        #     name=config.embedder_model_name, torch_dtype=config.embedder_torch_dtype
        # )
        embedder, embedder_tokenizer = None, None

        if config.auto_encoder_name == 'sentence-transformers/sentence-t5-base':
            embedder_encoder = T5ForConditionalGeneration.from_pretrained('sentence-transformers/sentence-t5-base').encoder
            tokenizer_t5 = T5Tokenizer.from_pretrained('sentence-transformers/sentence-t5-base')
            encoder_emb_len = 768
        elif config.auto_encoder_name == 'intfloat/e5-large': 
            embedder_encoder = AutoModel.from_pretrained('intfloat/e5-large')
            tokenizer_t5 = AutoTokenizer.from_pretrained('intfloat/e5-large')
            encoder_emb_len = 1024
        elif config.auto_encoder_name == 'sentence-transformers/sentence-t5-large':
            embedder_encoder = T5ForConditionalGeneration.from_pretrained('sentence-transformers/sentence-t5-large').encoder
            tokenizer_t5 = T5Tokenizer.from_pretrained('sentence-transformers/sentence-t5-large')
            encoder_emb_len = 1024
        else:
            raise ValueError('please configure the name of the encoder.')
        print(f'load weight from {config.auto_encoder_name}')
        embedder_encoder.to('cuda:0')
        
        tokenizer = load_tokenizer(
            # config.model_name_or_path,
            't5-base',
            max_length=config.max_seq_length,
        )
        num_repeat_tokens = config.num_repeat_tokens
        embedder_no_grad = config.embedder_no_grad

        self.encoder_decoder = encoder_decoder  # .to_bettertransformer()
        ######################################################
        self.num_repeat_tokens = num_repeat_tokens

        self.embedder_is_decoder = False

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        # if embedder_model_api:
        #     assert use_frozen_embeddings_as_input, "must precompute embeddings w/ api"
        #     # Hard-code OpenAI embedding dim
        #     self.embedder_dim = 1536
        #     bottleneck_dim = self.embedder_dim
        # elif isinstance(embedder, SentenceTransformer):
        #     self.embedder_dim = embedder.get_sentence_embedding_dimension()
        #     bottleneck_dim = self.embedder_dim
        # else:
        #     self.embedder_dim = embedder.config.hidden_size
        #     bottleneck_dim = self.embedder_dim
        self.embedder_no_grad = embedder_no_grad
        self.use_frozen_embeddings_as_input = use_frozen_embeddings_as_input
        # self.bottleneck_dim = bottleneck_dim
        self.embedder_dim = encoder_emb_len # hardcode the dimension
        bottleneck_dim = 768

        self.embedding_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),  # TODO consider dropout or normalization here.
            nn.Linear(bottleneck_dim, encoder_hidden_dim * num_repeat_tokens),
        )
        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)
        ######################################################
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.tokenizer_t5 = tokenizer_t5
        self.embedder_encoder = embedder_encoder
        self.embedder_tokenizer = tokenizer_t5
        self.embedder_model_api = embedder_model_api
        # self.freeze(freeze_strategy=config.freeze_strategy)
        self.embedder_fake_with_zeros = embedder_fake_with_zeros

        self.embedding_transform_strategy = "repeat"  # "none" # "repeat"
        self.embeddings_from_layer_n = embeddings_from_layer_n
        self.noise_level = 0
        self.encoder_decoder.decoder_start_token_id = 0


        # self.graph_dataset=PygNodePropPredDataset(name='ogbn-arxiv', transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]))
        # self.graph_data = self.graph_dataset[0]

        # if config.dataset_name == 'hubin/arxiv_title_abstract_all_for_train':
        #     feature_emb = torch.load('feature_emb_title_abs.pt')
        #     print('load embedding from feature_emb_title_abs.pt')
        # else: 
        #     feature_emb = torch.load('feature_emb.pt')
        #     print('load embedding from feature_emb.pt')

        # self.graph_data.x = torch.tensor(feature_emb)
        
        # self.graph_data = self.graph_data.to('cuda')
        # self.graph_data.x = self.graph_data.x.to(torch.bfloat16)
        # self.graph_data.adj_t = self.graph_data.adj_t.to(torch.bfloat16)

        # num_sage_layers = config.num_sage_layers

        # self.graph_model = SAGE_embedding(self.graph_data.num_features, 1024, 1024, 3, 0.5, num_sage_layers).to('cuda:0')
        # self.graph_model = self.graph_model.to(torch.bfloat16)

        # self.sage_layer = nn.Sequential(nn.Linear(encoder_emb_len*2, 1024))
        # self.sage_layer = nn.Sequential(nn.Linear(768*2, 1024))


        


    
        
        # self.id2text_map = torch.load('arxiv_title_labels_matrix.pt')

        # self.data_count = 0



    def _freeze_encoder(self):
        freeze_params(self.encoder_decoder.encoder)

    def _freeze_decoder(self):
        # github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L1229-L1231
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)

    def freeze(self, freeze_strategy: str):
        assert freeze_strategy in FREEZE_STRATEGIES

        if freeze_strategy == "decoder":
            self._freeze_decoder()
        elif freeze_strategy == "encoder":
            self._freeze_encoder()
        elif freeze_strategy == "encoder_and_decoder":
            self._freeze_encoder()
            self._freeze_decoder()
            # in this case, freeze embeddings too
            freeze_params(self.encoder_decoder.shared)
        elif freeze_strategy == "none":
            pass
        else:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device

    def _process_embedder_output(
        self,
        outputs: transformers.modeling_outputs.BaseModelOutput,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and (outputs.pooler_output is not None):
            return outputs.pooler_output
        else:
            if self.embeddings_from_layer_n is not None:
                assert hasattr(
                    outputs, "hidden_states"
                ), "output missing hidden states - did you remember to initialize the model with output_hidden_states=True?"
                hidden_state = outputs.hidden_states[self.embeddings_from_layer_n]
                embeddings = mean_pool(hidden_state, attention_mask)
            else:
                hidden_state = outputs.last_hidden_state
                embeddings = mean_pool(hidden_state, attention_mask)
            return embeddings

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # token_type_ids: Optional[torch.Tensor] = None, # not used
    ) -> torch.Tensor:
        embedder = self.embedder
        # print("** call_embedding_model")
        if self.embedder_no_grad:
            embedder.eval()

        if self.embedder_fake_with_zeros:
            batch_size = input_ids.shape[0]
            return torch.zeros(
                (batch_size, self.embedder_dim),
                dtype=torch.float32,
                device=self.embedder_device,
            )
        elif self.embedder_model_api:
            embeddings = embed_api(
                input_ids=input_ids,
                embedder_tokenizer=self.embedder_tokenizer,
                api_name=self.embedder_model_api,
            )
        elif isinstance(self.embedder, SentenceTransformer):
            # sentence-transformers is kind of really annoying
            model_output = embedder(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            embeddings = model_output["sentence_embedding"]
        else:
            model_output = embedder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self._process_embedder_output(model_output, attention_mask)

        if self.noise_level > 0:
            embeddings += self.noise_level * torch.randn(
                embeddings.shape, device=embeddings.device
            )
        return embeddings

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("** embed_and_project")
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))
        # if frozen_embeddings is not None:
        embeddings = frozen_embeddings
        # print(embeddings.shape)
        assert len(embeddings.shape) == 2  # batch by d
        # elif self.embedder_no_grad:
        #     with torch.no_grad():
        #         embeddings = self.call_embedding_model(
        #             input_ids=embedder_input_ids,
        #             attention_mask=embedder_attention_mask,
        #         )
        # else:
        #     embeddings = self.call_embedding_model(
        #         input_ids=embedder_input_ids,
        #         attention_mask=embedder_attention_mask,
        #     )
        if self.embedding_transform_strategy == "repeat":
            repeated_embeddings = self.embedding_transform(embeddings)
            # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
            embeddings = repeated_embeddings.reshape(
                (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
            )
        elif self.embedding_transform_strategy == "nearest_neighbors":
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError(
                f"unknown embedding transformation strategy {self.embedding_transform_strategy}"
            )
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )
        return embeddings, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit

        node_idx = generation_kwargs.get('node_idx')

        # title_abs_list_1st_order = []
        # title_abs_list_2nd_order = []
        # second_order_nb_list = []
        # for idx_ in node_idx:
        #     idx_int = idx_.item()
        #     sampled_neighbours_ids = random.sample(adj_list[idx_int], min(num_sampled_neighbour, len(adj_list[idx_int])))
        #     for nb_ids in sampled_neighbours_ids:
        #         second_order_nb_list += adj_list[nb_ids]
        #     neighbour_idx_1st_order = random.choice(adj_list[idx_int])
        #     neighbour_idx_2nd_order = random.choice(second_order_nb_list[idx_int])
        #     paperid = df_node2paper[df_node2paper['node idx'] == neighbour_idx_1st_order]['paper id'].values[0]
        #     title = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['title'].values[0]
        #     abstract = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['abstract'].values[0]
        #     title_abs_list_1st_order.append('title: ' + title + '; abstract: ' + abstract)

        #     paperid = df_node2paper[df_node2paper['node idx'] == neighbour_idx_2nd_order]['paper id'].values[0]
        #     title = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['title'].values[0]
        #     abstract = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['abstract'].values[0]
        #     title_abs_list_2nd_order.append('title: ' + title + '; abstract: ' + abstract)

        # input_ids_title_list_nb = self.tokenizer_t5(title_abs_list_1st_order, padding=True, truncation=True, return_tensors="pt" ,max_length=encoder_input_token_max_len).to('cuda')
        # sent_embs_last_state_nb_1st_order = self.embedder_encoder(input_ids=input_ids_title_list_nb['input_ids'], attention_mask=input_ids_title_list_nb['attention_mask']).last_hidden_state # (#neighbour, 768)
        # sent_embs_nb_1st_order = average_pool(sent_embs_last_state_nb_1st_order, input_ids_title_list_nb['attention_mask'])

        # input_ids_title_list_nb = self.tokenizer_t5(title_abs_list_2nd_order, padding=True, truncation=True, return_tensors="pt" ,max_length=encoder_input_token_max_len).to('cuda')
        # sent_embs_last_state_nb_2nd_order = self.embedder_encoder(input_ids=input_ids_title_list_nb['input_ids'], attention_mask=input_ids_title_list_nb['attention_mask']).last_hidden_state # (#neighbour, 768)
        # sent_embs_nb_2nd_order = average_pool(sent_embs_last_state_nb_2nd_order, input_ids_title_list_nb['attention_mask'])

        # self.positive_samples_1st_order =  sent_embs_nb_1st_order # (batch size, 768)
        # self.positive_samples_2st_order =  sent_embs_nb_2nd_order # (batch size, 768)

        title_list = []
        for idx_ in node_idx:
            idx_int = idx_.item()
            paperid = df_node2paper[df_node2paper['node idx'] == idx_int]['paper id'].values[0]
            title = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['title'].values[0]
            abstract = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['abstract'].values[0]
            title_list.append('title: ' + title + '; abstract: ' + abstract)
        input_ids_title_list = self.tokenizer_t5(title_list, padding=True, truncation=True, return_tensors="pt" ,max_length=encoder_input_token_max_len).to('cuda')
        center_sent_embs_batch_last_hidden_state = self.embedder_encoder(input_ids=input_ids_title_list['input_ids'], attention_mask=input_ids_title_list['attention_mask']).last_hidden_state
        center_sent_embs_batch = average_pool(center_sent_embs_batch_last_hidden_state, input_ids_title_list['attention_mask'])
        
        self.query = center_sent_embs_batch

        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=inputs.get("embedder_input_ids"),
            embedder_attention_mask=inputs.get("embedder_attention_mask"),
            frozen_embeddings=center_sent_embs_batch,
        )

        generation_kwargs.pop("node_idx")
        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                decoder_start_token_id = 0,
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_start_token_id = 0,
                **generation_kwargs,
            )

    def forward(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        node_idx = kwargs.get('node_idx')

        title_abs_list_1st_order = []
        title_abs_list_2nd_order = []
        second_order_nb_list = []
        for idx_ in node_idx:
            idx_int = idx_.item()
            sampled_neighbours_ids = random.sample(self.adj_list[idx_int], min(num_sampled_neighbour, len(self.adj_list[idx_int])))
            for nb_ids in sampled_neighbours_ids:
                second_order_nb_list += self.adj_list[nb_ids]
            neighbour_idx_1st_order = random.choice(self.adj_list[idx_int])
            neighbour_idx_2nd_order = random.choice(second_order_nb_list)
            paperid = df_node2paper[df_node2paper['node idx'] == neighbour_idx_1st_order]['paper id'].values[0]
            title = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['title'].values[0]
            abstract = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['abstract'].values[0]
            title_abs_list_1st_order.append('title: ' + title + '; abstract: ' + abstract)

            paperid = df_node2paper[df_node2paper['node idx'] == neighbour_idx_2nd_order]['paper id'].values[0]
            title = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['title'].values[0]
            abstract = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['abstract'].values[0]
            title_abs_list_2nd_order.append('title: ' + title + '; abstract: ' + abstract)

        input_ids_title_list_nb = self.tokenizer_t5(title_abs_list_1st_order, padding=True, truncation=True, return_tensors="pt" ,max_length=encoder_input_token_max_len).to('cuda')
        sent_embs_last_state_nb_1st_order = self.embedder_encoder(input_ids=input_ids_title_list_nb['input_ids'], attention_mask=input_ids_title_list_nb['attention_mask']).last_hidden_state # (#neighbour, 768)
        sent_embs_nb_1st_order = average_pool(sent_embs_last_state_nb_1st_order, input_ids_title_list_nb['attention_mask'])

        input_ids_title_list_nb = self.tokenizer_t5(title_abs_list_2nd_order, padding=True, truncation=True, return_tensors="pt" ,max_length=encoder_input_token_max_len).to('cuda')
        sent_embs_last_state_nb_2nd_order = self.embedder_encoder(input_ids=input_ids_title_list_nb['input_ids'], attention_mask=input_ids_title_list_nb['attention_mask']).last_hidden_state # (#neighbour, 768)
        sent_embs_nb_2nd_order = average_pool(sent_embs_last_state_nb_2nd_order, input_ids_title_list_nb['attention_mask'])

        self.positive_samples_1st_order =  sent_embs_nb_1st_order # (batch size, 768)
        self.positive_samples_2st_order =  sent_embs_nb_2nd_order # (batch size, 768)

        title_list = []
        for idx_ in node_idx:
            idx_int = idx_.item()
            paperid = df_node2paper[df_node2paper['node idx'] == idx_int]['paper id'].values[0]
            title = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['title'].values[0]
            abstract = df_paperid2titleabs[df_paperid2titleabs['paperid_int'] == paperid]['abstract'].values[0]
            title_list.append('title: ' + title + '; abstract: ' + abstract)
        input_ids_title_list = self.tokenizer_t5(title_list, padding=True, truncation=True, return_tensors="pt" ,max_length=encoder_input_token_max_len).to('cuda')
        center_sent_embs_batch_last_hidden_state = self.embedder_encoder(input_ids=input_ids_title_list['input_ids'], attention_mask=input_ids_title_list['attention_mask']).last_hidden_state
        center_sent_embs_batch = average_pool(center_sent_embs_batch_last_hidden_state, input_ids_title_list['attention_mask'])
        
        self.query = center_sent_embs_batch
        
        # Unused: input_ids, attention_mask
        inputs_embeds, attention_mask = self.embed_and_project(
            embedder_input_ids=embedder_input_ids,
            embedder_attention_mask=embedder_attention_mask,
            frozen_embeddings=center_sent_embs_batch,
        )

        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )

