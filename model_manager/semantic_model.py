import torch

from model_manager.hsg_model import HSumGraph
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from torch import nn


class GraphAttentionPoolModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphAttentionPoolModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')
        self.pooling = GlobalAttentionPooling(nn.Linear(out_feats, 1))

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        # Apply graph attention pooling
        pooled_features = self.pooling(g, x)
        return pooled_features


class SemanticModel(HSumGraph):
    def __init__(self, hps, embed):
        self.syntax_generate_graph_threshold = 0.7
        super(SemanticModel, self).__init__(hps=hps, embed=embed)
        self.semantic_gnn = GraphAttentionPoolModel(in_feats=300, hidden_feats=128, out_feats=64)
        self.sentence_classifier = nn.Linear(in_features=128, out_features=2)
        self.to(self._hps.device)

    def sentence_semantic_graph(self, graph):
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        words = graph.nodes[snode_id].data["words"]
        words_feature = self._embed(words)
        graphs = []
        for i, sentence_word_feature in enumerate(words_feature):
            sentence = sentence_word_feature[:words[i].nonzero().size(0), :]
            cos_sim = F.cosine_similarity(sentence.unsqueeze(1), sentence, dim=2)
            rows, cols = (cos_sim > self.syntax_generate_graph_threshold).nonzero(as_tuple=True)
            g = dgl.graph((rows, cols), num_nodes=len(sentence))
            g.nodes[:].data["word_feature"] = sentence
            graphs.append(g)
        return dgl.batch(graphs=graphs)

    def forward(self, graph):

        word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [snode, n_feature_size]
        semantic_graph = self.sentence_semantic_graph(graph=graph)
        semantic_feature = self.semantic_gnn(semantic_graph, semantic_graph.nodes[:].data["word_feature"])

        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        # result = self.wh(sent_state)
        hsg_feature = sent_state
        sentence_feature = torch.concat([semantic_feature, hsg_feature], dim=-1)
        return self.sentence_classifier(sentence_feature)
#

# class SemanticModel(nn.Module):
#     def __init__(self, hps, embed):
#         super(SemanticModel, self).__init__()
#
#         self.semantic_generate_graph_threshold = 0.7
#         self._hps = hps
#         self._embed = embed
#         self.semantic_gnn = GraphAttentionPoolModel(in_feats=300, hidden_feats=128, out_feats=64)
#         self.sentence_classifier = nn.Linear(in_features=64, out_features=2)
#         self.to(self._hps.device)
#
#     def sentence_semantic_graph(self, graph):
#         snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
#         words = graph.nodes[snode_id].data["words"]
#         words_feature = self._embed(words)
#         graphs = []
#         for i, sentence_word_feature in enumerate(words_feature):
#             sentence = sentence_word_feature[:words[i].nonzero().size(0), :]
#             cos_sim = F.cosine_similarity(sentence.unsqueeze(1), sentence, dim=2)
#             rows, cols = (cos_sim > self.semantic_generate_graph_threshold).nonzero(as_tuple=True)
#             g = dgl.graph((rows, cols), num_nodes=len(sentence))
#             g.nodes[:].data["word_feature"] = sentence
#             graphs.append(g)
#         return dgl.batch(graphs=graphs)
#
#     def forward(self, graph):
#
#         semantic_graph = self.sentence_semantic_graph(graph=graph)
#         semantic_feature = self.semantic_gnn(semantic_graph, semantic_graph.nodes[:].data["word_feature"])
#
#         return self.sentence_classifier(semantic_feature)
#
# class SemanticModel(HSumGraph):
#     def __init__(self, hps, embed):
#         self.semantic_generate_graph_threshold = 0.7
#         super(SemanticModel, self).__init__(hps=hps, embed=embed)
#         self.semantic_gnn = GraphAttentionPoolModel(in_feats=300, hidden_feats=128, out_feats=64)
#         self.fusion_linear = torch.nn.Sequential(torch.nn.Linear(in_features=128, out_features=64), torch.nn.ReLU())
#         self.sentence_classifier = nn.Linear(in_features=64, out_features=2)
#         self.to(self._hps.device)
#
#     def sentence_semantic_graph(self, graph):
#         snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
#         words = graph.nodes[snode_id].data["words"]
#         words_feature = self._embed(words)
#         graphs = []
#         for i, sentence_word_feature in enumerate(words_feature):
#             sentence = sentence_word_feature[:words[i].nonzero().size(0), :]
#             cos_sim = F.cosine_similarity(sentence.unsqueeze(1), sentence, dim=2)
#             rows, cols = (cos_sim > self.semantic_generate_graph_threshold).nonzero(as_tuple=True)
#             g = dgl.graph((rows, cols), num_nodes=len(sentence))
#             g.nodes[:].data["word_feature"] = sentence
#             graphs.append(g)
#         return dgl.batch(graphs=graphs)
#
#     def forward(self, graph):
#
#         word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]
#
#         sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [snode, n_feature_size]
#         semantic_graph = self.sentence_semantic_graph(graph=graph)
#         semantic_feature = self.semantic_gnn(semantic_graph, semantic_graph.nodes[:].data["word_feature"])
#         sent_feature = self.fusion_linear(torch.cat((semantic_feature, sent_feature),dim=-1))
        # the start state
        # word_state = word_feature
        # sent_state = self.word2sent(graph, word_feature, sent_feature)
        #
        # for i in range(self._n_iter):
        #     sent -> word
            # word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            # sent_state = self.word2sent(graph, word_state, sent_state)

        # result = self.wh(sent_state)
        # hsg_feature = sent_state
        # sentence_feature = torch.concat([semantic_feature, hsg_feature], dim=-1)
        # return self.sentence_classifier(sent_state)
