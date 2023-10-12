import torch

from model_manager.hsg_model import HSumGraph
from dgl.nn.pytorch import GATConv
import dgl


class CompletedSentenceGraph(torch.nn.Module):
    def __init__(self, hps):
        super(CompletedSentenceGraph, self).__init__()
        self.hps = hps
        self.num_heads = 4
        self.s2s_gat_conv = GATConv(in_feats=hps.hidden_size, out_feats=hps.hidden_size, num_heads=self.num_heads)

    def make_sentence_graph(self, graph):
        new_graphs = []

        graphs = dgl.unbatch(graph)
        for g in graphs:
            sentences = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            nodes = len(sentences)
            x, y = torch.meshgrid(torch.arange(nodes), torch.arange(nodes))
            new_graphs.append(dgl.graph((x.flatten(), y.flatten())))

        return dgl.batch(new_graphs)

    def forward(self, graph, sent_feature):
        sentence_graph = self.make_sentence_graph(graph).to(self.hps.device)
        sent_features = self.s2s_gat_conv(sentence_graph, sent_feature)
        return sent_features.mean(dim=1)


class HSGModel(HSumGraph):
    def __init__(self, hps, embed):
        super(HSGModel, self).__init__(hps=hps, embed=embed)
        self.completed_graph_layer = CompletedSentenceGraph(hps=hps)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hps.hidden_size, hps.hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hps.hidden_size, 1),
            torch.nn.Sigmoid()
        )
        self.to(hps.device)

    def forward(self, graph):
        word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [snode, n_feature_size]

        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        sent_state2 = self.completed_graph_layer(graph, sent_feature)

        # result = self.wh(sent_state)

        return self.classifier(torch.cat([sent_state, sent_state2], dim=1))
