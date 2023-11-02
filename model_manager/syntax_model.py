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


class GNNLayer(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GNNLayer, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, 'mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, 'mean')

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x


class SyntaxModel(HSumGraph):
    def __init__(self, hps, embed):
        # self.syntax_generate_graph_threshold = 0.7
        super(SyntaxModel, self).__init__(hps=hps, embed=embed)
        self.sentence_classifier = nn.Linear(in_features=128, out_features=2)
        self.syntax_gnn = GNNLayer(in_feats=300,hidden_feats=150,out_feats=64)
        self.syntax_pooling = GlobalAttentionPooling(nn.Linear(64, 1))
        self.to(self._hps.device)

    def forward(self, G):
        graph, syntax_graph = G
        graph = graph.to(self._hps.device)
        syntax_graph = syntax_graph.to(self._hps.device)

        word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [snode, n_feature_size]
        syntax_word_feature = self.syntax_gnn(syntax_graph, self._embed(syntax_graph.ndata["word_id"]))

        syntax_graph_list = []
        for text_graph in dgl.unbatch(syntax_graph):
            unique_sent_ids = text_graph.ndata["sent_id"].unique()
            # unique_sent_ids = text_graph.nodes[:].data["sent_id"].unique()
            for sent_id in unique_sent_ids:
                g = text_graph.subgraph(text_graph.filter_nodes(lambda nodes: nodes.data["sent_id"] == sent_id))
                syntax_graph_list.append(g)
        syntax_sent_feature = self.syntax_pooling(dgl.batch(syntax_graph_list),syntax_word_feature)

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
        sentence_feature = torch.concat([syntax_sent_feature, hsg_feature], dim=-1)
        return self.sentence_classifier(sentence_feature)
