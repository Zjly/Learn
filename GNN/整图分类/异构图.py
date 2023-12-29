import dgl.nn.pytorch as dglnn
import torch.nn as nn
import dgl
import torch

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs是节点的特征
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # 通过平均读出值来计算单图的表征
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)

# import dgl.data
# dataset = dgl.data.GINDataset('MUTAG', False)
#
# from dgl.dataloading import GraphDataLoader
# dataloader = GraphDataLoader(
#     dataset,
#     batch_size=1024,
#     drop_last=False,
#     shuffle=True)
#

# import torch.nn.functional as F
#
# # etypes是一个列表，元素是字符串类型的边类型
# model = HeteroClassifier(10, 20, 5, etypes)
# opt = torch.optim.Adam(model.parameters())
# for epoch in range(20):
#     for batched_graph, labels in dataloader:
#         logits = model(batched_graph)
#         loss = F.cross_entropy(logits, labels)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()