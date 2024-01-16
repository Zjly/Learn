import os

import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from collections import Counter


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

        # 打印图的信息
        # print(graph)
        print(api_sequence)
        print(graph.edges())
        print(graph.edata['weight'])
        print(graph.ndata['feature'])
        print("_________________________________")

    api_dataset = APIDataset(graph_list, label_list)

    train_loader = GraphDataLoader(
        api_dataset,
        batch_size=2,
        pin_memory=torch.cuda.is_available(),
    )

    print(1)


def train(train_loader, val_loader, device, model):
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
            feat = batched_graph.ndata.pop("attr")
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc = evaluate(train_loader, device, model)
        valid_acc = evaluate(val_loader, device, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                epoch, total_loss / (batch + 1), train_acc, valid_acc
            )
        )

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
