import torch
import torch.nn as nn
import dgl
from model_manager.hsg_model import HSumGraph


class RnnHSGModel(HSumGraph):
    def __init__(self, hps, embed):
        super(RnnHSGModel, self).__init__(hps=hps, embed=embed)
        self.rnn = torch.nn.GRU(hps.hidden_size, hps.hidden_size, 4, bias=True, batch_first=True, bidirectional=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hps.hidden_size, hps.hidden_size),
            torch.nn.LeakyReLU(),
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

        graph_list = dgl.unbatch(graph)
        indices = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_list]
        rnn_results = torch.Tensor().to(self._hps.device)
        for sentence_vector in torch.split(sent_feature, indices):
            rnn_result = self.rnn(sentence_vector.unsqueeze(dim=0))[0].squeeze()
            rnn_results = torch.cat([rnn_results, rnn_result])
        result = self.classifier(rnn_results)

        return result
        # return self.sentence_level_model(sent_features, probabilities)
