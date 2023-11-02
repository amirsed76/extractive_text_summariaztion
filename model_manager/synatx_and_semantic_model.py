import torch
from model_manager.hsg_model import HSumGraph
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from torch import nn
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab, token_match=str.split)


class SentenceGraphModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(SentenceGraphModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')
        self.pool = GlobalAttentionPooling(nn.Linear(64, 1))

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return self.pool(g, x)


class HSGModel(HSumGraph):
    def forward(self, graph, word_feature, sent_feature):
        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        # result = self.wh(sent_state)
        return sent_state

    def set_snfeature(self, graph, snode_id):
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        features, glen = self.get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature


class SemanticSyntaxHSGModel(nn.Module):
    def __init__(self, hps, embed):
        super(SemanticSyntaxHSGModel, self).__init__()
        self.semantic_generate_graph_threshold = 0.7
        self.embed = embed

        self.hps = hps
        self.hsg_layer = HSGModel(hps=hps, embed=embed)
        self.syntax_layer = SentenceGraphModel(in_feats=300, hidden_feats=150, out_feats=64)
        self.semantic_layer = SentenceGraphModel(in_feats=300, hidden_feats=150, out_feats=64)
        self.sentence_classifier = torch.nn.Sequential(torch.nn.Linear(3 * 64, 128), torch.nn.ReLU(),
                                                       torch.nn.Linear(128, 2))
        self.to(self.hps.device)

    def get_syntax_graph(self, syntax_graph):
        syntax_graph_list = []
        for text_graph in dgl.unbatch(syntax_graph):
            unique_sent_ids = text_graph.ndata["sent_id"].unique()
            # unique_sent_ids = text_graph.nodes[:].data["sent_id"].unique()
            for sent_id in unique_sent_ids:
                g = text_graph.subgraph(text_graph.filter_nodes(lambda nodes: nodes.data["sent_id"] == sent_id))
                syntax_graph_list.append(g)

        return dgl.batch(syntax_graph_list)

    def get_semantic_graph(self, graph, snode_id):
        words = graph.nodes[snode_id].data["words"]
        words_feature = self.embed(words)
        graphs = []
        for i, sentence_word_feature in enumerate(words_feature):
            sentence = sentence_word_feature[:words[i].nonzero().size(0), :]
            cos_sim = F.cosine_similarity(sentence.unsqueeze(1), sentence, dim=2)
            rows, cols = (cos_sim > self.semantic_generate_graph_threshold).nonzero(as_tuple=True)
            g = dgl.graph((rows, cols), num_nodes=len(sentence))
            g.nodes[:].data["word_feature"] = sentence
            graphs.append(g)
        return dgl.batch(graphs=graphs)

    def forward(self, G):
        hsg_graph, syntax_graph = G
        snode_id = hsg_graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        word_feature = self.hsg_layer.set_wnfeature(hsg_graph)  # [wnode, embed_size]
        sent_feature = self.hsg_layer.n_feature_proj(
            self.hsg_layer.set_snfeature(hsg_graph, snode_id))  # [snode, n_feature_size]
        hsg_feature = self.hsg_layer(hsg_graph, word_feature, sent_feature)
        syntax_word_feature = self.embed(syntax_graph.ndata["word_id"])
        syntax_feature = self.syntax_layer(self.get_syntax_graph(syntax_graph=syntax_graph), syntax_word_feature)
        semantic_graph = self.get_semantic_graph(hsg_graph, snode_id)
        semantic_feature = self.semantic_layer(semantic_graph, semantic_graph.nodes[:].data["word_feature"])
        sentence_feature = torch.concat([semantic_feature, syntax_feature, hsg_feature], dim=-1)
        return self.sentence_classifier(sentence_feature)
