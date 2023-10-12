import dgl
import torch
from model_manager.hsg_model import HSumGraph


class HSGModel(HSumGraph):
    def __init__(self, hps, embed):
        super(HSGModel, self).__init__(hps=hps, embed=embed)
        self.embed_linear = torch.nn.Sequential(torch.nn.Linear(300, 300, bias=True), torch.nn.ReLU())
        self.word_classifier = torch.nn.Linear(300, 2)

        # self.rnn = torch.nn.GRU(hps.hidden_size, hps.hidden_size, 2, bias=True, batch_first=True,
        #                         bidirectional=True)

        self.sentence_classifier = torch.nn.Sequential(torch.nn.Linear(self.n_feature, 32),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(32, 1),
                                                       torch.nn.Sigmoid())

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

        # graph_list = dgl.unbatch(graph)
        # indices = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_list]
        # rnn_results = torch.Tensor().to(self._hps.device)
        # for sentence_vector in torch.split(sent_feature, indices):
        #     rnn_result = self.rnn(sentence_vector.unsqueeze(dim=0))[0].squeeze()
        #     rnn_results = torch.cat([rnn_results, rnn_result])
        sentence_result = self.sentence_classifier(sent_state)
        word_result = self.word_classifier(word_state)
        return sentence_result, word_result

    def embed(self, word_id):
        x = self._embed(word_id)
        x = self.embed_linear(x)
        return x

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
