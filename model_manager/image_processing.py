import dgl
import torch

from model_manager.hsg_model import HSumGraph


class HSGModel(HSumGraph):
    def __init__(self, hps, embed):
        super(HSGModel, self).__init__(hps=hps, embed=embed)
        self.embed_linear = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(300, 300, bias=True))
        self.word_classifier = torch.nn.Linear(300, 2)
        self.sentence_rnn = torch.nn.GRU(hps.hidden_size * 2, hps.hidden_size, 4, bias=True, batch_first=True,
                                         bidirectional=False)

        self.sentence_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.n_feature, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid())

        self.to(hps.device)

    def sentence_classify(self, graph, sent_state):
        graph_list = dgl.unbatch(graph)
        result = torch.Tensor().to(self._hps.device)
        sentence_lengths = []
        index = 0
        for g in graph_list:
            s_node_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            sentence_len = len(s_node_id)
            sentence_lengths.append(sentence_len)
            result = torch.cat([result, sent_state[index:index + sentence_len, :]])
            index = index + sentence_len
            if sentence_len < self._hps.doc_max_timesteps:
                result = torch.cat(
                    [result, torch.zeros(self._hps.doc_max_timesteps - sentence_len, 2 * self.n_feature).to(
                        self._hps.device)])

        text_feature = result.reshape(-1, self._hps.doc_max_timesteps, self.n_feature * 2)
        text_feature = self.sentence_rnn(text_feature)[0]
        sentence_features = torch.Tensor().to(self._hps.device)
        for i in range(text_feature.shape[0]):
            sentence_features = torch.cat([sentence_features, text_feature[i, :sentence_lengths[i], :]])

        result = self.sentence_classifier(sentence_features)

        return result

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

        sentence_result = self.sentence_classify(graph=graph, sent_state=torch.cat([sent_state, sent_feature], dim=1))
        word_result = self.word_classifier(word_feature)
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
