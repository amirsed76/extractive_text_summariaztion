import dgl
import torch

from model_manager.hsg_model import HSumGraph
from dgl.nn.pytorch import GCN2Conv, GATConv
import networkx as nx


class HSGModel(HSumGraph):
    def __init__(self, hps, embed):
        super(HSGModel, self).__init__(hps=hps, embed=embed)
        self.embed_linear = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(300, 300, bias=True))
        self.word_classifier = torch.nn.Linear(300, 2)

        # self.rnn = torch.nn.GRU(hps.hidden_size, hps.hidden_size, 2, bias=True, batch_first=True,
        #                         bidirectional=True)
        self.s2s_gat_conv = GATConv(in_feats=hps.hidden_size, out_feats=hps.hidden_size, num_heads=3)

        self.sentence_classifier = torch.nn.Sequential(torch.nn.Linear(3 * self.n_feature, 32),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(32, 2))

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

        sentence_graph = self.make_sentence_graph(graph).to(self._hps.device)
        sent_features = self.s2s_gat_conv(sentence_graph, sent_state)
        sent_features = sent_features.reshape(-1, self.n_feature * 3)
        sentence_result = self.sentence_classifier(sent_features)
        word_result = self.word_classifier(word_feature)
        return sentence_result, word_result

    def embed(self, word_id):
        x = self._embed(word_id)
        x = self.embed_linear(x)
        return x

    @staticmethod
    def make_sentence_graph(graph):
        # last_index = 0

        result_graphs = []

        graphs = dgl.unbatch(graph)
        for g in graphs:
            sentences = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            n = sentences.shape[0]
            nx_graph = nx.complete_graph(n)
            dgl_graph = dgl.from_networkx(nx_graph)
            result_graphs.append(dgl_graph)

        return dgl.batch(result_graphs)

        # for g in graphs:
        #     sentences = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        #     new_u = torch.Tensor(list(range(len(sentences) - 1))) + last_index
        #     new_v = torch.Tensor(list(range(1, len(sentences)))) + last_index
        #     u = torch.cat([u, new_u, new_v])
        #     v = torch.cat([v, new_v, new_u])
        #     last_index += len(sentences)

        # return dgl.graph((list(u), list(v)))

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)  # for word to supernode(sent&doc)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self.embed(wid)  # [n_wnodes, D]
        # w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)

        return w_embed
