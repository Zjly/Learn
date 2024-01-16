from collections import Counter

import dgl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer

def create_graph():
    df = pd.read_feather('dataset_100.feather')
    api_count = 13053
    hidden_size = 768
    feature = torch.randn(api_count, hidden_size)

    graph_list = []
    label_list = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        api_sequence = row['api_sequence']
        label = row['label']

        if len(api_sequence) <= 1:
            continue

        # 创建有向图
        graph = dgl.DGLGraph()

        # 添加节点
        graph.add_nodes(api_count)

        # 计算边的权重（即出现次数）
        edge_counter = Counter(zip(api_sequence, api_sequence[1:]))
        edge_weights = torch.tensor(list(edge_counter.values()), dtype=torch.float32).view(-1, 1)

        # 添加带有权重的边
        src, dst = zip(*edge_counter.keys())
        graph.add_edges(src, dst)

        # 将权重作为边的特征
        graph.edata['weight'] = edge_weights.view(-1, 1)
        graph.ndata['feature'] = feature

        graph_list.append(graph)
        label_list.append(label)

        # # 打印图的信息
        # print(api_sequence)
        # print(graph.edges())
        # print(graph.edata['weight'])
        # print(graph.ndata['feature'])
        # print("_________________________________")

    api_dataset = APIDataset(graph_list, label_list)

    train_loader = GraphDataLoader(
        api_dataset,
        batch_size=8,
        pin_memory=torch.cuda.is_available(),
    )

    model = GIN(768, 768, 13053)
    train(train_loader, torch.device('cpu'), model)


def train(train_loader, device, model):
    # loss function, optimizer and scheduler
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # training loop
    for epoch in range(350):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feat = batched_graph.ndata.pop("feature")
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(total_loss / (batch + 1))


class APIDataset(DGLDataset):
    def __init__(self, graph_list, label_list, url=None, raw_dir=None, save_dir=None, force_reload=False,
                 verbose=False):
        super(APIDataset, self).__init__(name='API_dataset',
                                         url=url,
                                         raw_dir=raw_dir,
                                         save_dir=save_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)
        self.graph_list = graph_list
        self.label_list = label_list

    def process(self):
        pass

    def __getitem__(self, idx):
        return self.graph_list[idx], self.label_list[idx]

    def __len__(self):
        return len(self.graph_list)

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass


if __name__ == '__main__':
    create_graph()
