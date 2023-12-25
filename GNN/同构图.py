import dgl
import torch

dataset = dgl.data.CiteseerGraphDataset()
graph = dataset[0]

# 构建一个2层的GNN模型
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

node_features = graph.ndata['feat']
node_labels = graph.ndata['label']
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
print(graph)
opt = torch.optim.Adam(model.parameters())

for epoch in range(100):
    model.train()
    # 使用所有节点(全图)进行前向传播计算
    logits = model(graph, node_features)
    # 计算损失值
    loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
    # 计算验证集的准确度
    acc = evaluate(model, graph, node_features, node_labels, valid_mask)
    # 进行反向传播计算
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())

    # 如果需要的话，保存训练好的模型。本例中省略。


