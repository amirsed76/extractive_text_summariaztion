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


class SyntaxModel(HSumGraph):
    def __init__(self, hps, embed):
        self.syntax_generate_graph_threshold = 0.7
        super(SyntaxModel, self).__init__(hps=hps, embed=embed)
        self.syntax_gnn = GraphAttentionPoolModel(in_feats=300, hidden_feats=128, out_feats=64)
        self.sentence_classifier = nn.Linear(in_features=128, out_features=2)
        self.to(self._hps.device)

    def sentence_syntax_graph(self, graph):
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        words = graph.nodes[snode_id].data["words"]
        words_feature = self._embed(words)
        graphs = []
        for s_index, s_id in enumerate(snode_id):
            text = " ".join([self._hps.vocab.id2word(w.item()) for w in words[s_index] if w.item() != 0])

            doc = nlp(text)
            g = dgl.graph(([token.i for token in doc], [token.head.i for token in doc]), num_nodes=len(doc)).to(
                self._hps.device)
            g.nodes[:].data["word_feature"] = words_feature[s_index, :len(doc), :]
            graphs.append(g)
        return dgl.batch(graphs=graphs)

    def forward(self, graph):

        word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [snode, n_feature_size]
        syntax_graph = self.sentence_syntax_graph(graph=graph)
        syntax_feature = self.syntax_gnn(syntax_graph, syntax_graph.nodes[:].data["word_feature"])

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
        sentence_feature = torch.concat([syntax_feature, hsg_feature], dim=-1)
        return self.sentence_classifier(sentence_feature)
