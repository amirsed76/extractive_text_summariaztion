import json

import torch
import dgl

import os
import pandas as pd
from tools.utils import eval_label
from tools.logger import *
import numpy as np


class TestPipLine:
    def __init__(self, model, m, test_dir, limited):
        """
            :param model: the model
            :param m: the number of sentence to select
            :param test_dir: for saving decode files
            :param limited: for limited Recall evaluation
        """
        self.model = model
        self.limited = limited
        self.m = m
        self.test_dir = test_dir
        self.extracts = []

        self.batch_number = 0
        self.running_loss = 0
        self.example_num = 0
        self.total_sentence_num = 0

        self._hyps = []
        self._refer = []

    def evaluation(self, G, index, valset):
        pass

    def getMetric(self):
        pass

    def SaveDecodeFile(self):
        import datetime
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 现在
        log_dir = os.path.join(self.test_dir, nowTime)
        with open(log_dir, "wb") as resfile:
            for i in range(self.rougePairNum):
                resfile.write(b"[Reference]\t")
                resfile.write(self._refer[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"[Hypothesis]\t")
                resfile.write(self._hyps[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"\n")
                resfile.write(b"\n")

    @property
    def running_avg_loss(self):
        return self.running_loss / self.batch_number

    @property
    def rougePairNum(self):
        return len(self._hyps)

    @property
    def hyps(self):
        if self.limited:
            hlist = []
            for i in range(self.rougePairNum):
                k = len(self._refer[i].split(" "))
                lh = " ".join(self._hyps[i].split(" ")[:k])
                hlist.append(lh)
            return hlist
        else:
            return self._hyps

    @property
    def refer(self):
        return self._refer

    @property
    def extractLabel(self):
        return self.extracts


def get_scores(self, G, indexes):
    scores = torch.Tensor()
    for g, index in zip(dgl.unbatch(G), indexes):
        snod_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)

        scores = torch.cat([scores, torch.Tensor(json.loads(self.scores[index])[:len(snod_id)])])

    return scores.to(self.hps.device)


class SLTester(TestPipLine):
    def __init__(self, model, m, hps, score_path, test_dir=None, limited=False, blocking_win=3):
        super().__init__(model, m, test_dir, limited)
        self.pred, self.true, self.match, self.match_true = 0, 0, 0, 0
        self._F = 0
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.Tensor([0.3, 0.7]).to(hps.device))
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        # self.criterion = CustomLoss()
        self.blocking_win = blocking_win
        self.score_path = score_path
        self.scores = pd.read_csv(score_path)["scores"].values.tolist()
        self.hps = hps
        self.outs = []

    def get_scores(self, G, indexes):
        scores = torch.Tensor()
        for g, index in zip(dgl.unbatch(G), indexes):
            snod_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)

            scores = torch.cat([scores, torch.Tensor(json.loads(self.scores[index])[:len(snod_id)])])

        return scores.to(self.hps.device)

    @staticmethod
    def torch_intersect(t1, t2, use_unique=False):
        return torch.tensor(np.intersect1d(t1.numpy(), t2.numpy()))

    def intersect_size(self, t1, t2):
        return np.intersect1d(t1, t2)

    def get_pred_idx__(self, g):
        # word_node_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        # word_ids = g.nodes[word_node_id].data["id"]
        # word_outs = g.ndata["word_out"][word_node_id]
        # important_words = g.ndata["id"][word_outs.argmax(dim=1).argwhere().reshape(-1)]
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        sentence_words = g.ndata["words"][snode_id]

        # sentence_words_np = sentence_words.cpu().numpy()
        # important_words_np = important_words.cpu().numpy()
        # pred_indexes = []
        # while len(important_words_np):
        #     sentence_scores = np.apply_along_axis(lambda row: self.intersect_size(row, important_words_np).shape[0], 1,
        #                                           sentence_words_np)
        #     index = sentence_scores.argmax()
        #     pred_indexes.append(index)
        #     important_words_np = np.setdiff1d(important_words_np, sentence_words_np[index])

        # sentence_scores = np.apply_along_axis(lambda row: self.intersect_size(row, important_words_np), 1,sentence_words_np)
        # top_k =[]
        # pred_idx = []
        sentence_scores = []
        for sentence in sentence_words:
            sentence_scores.append(len([w_id for w_id in sentence.tolist() if w_id in important_words]))
        topk, pred_idx = torch.topk(torch.Tensor(sentence_scores).cuda(), min(self.m, len(sentence_scores)))
        return topk, pred_idx

    def get_m_top(self, p_sent, N, m):
        # if m == 0:
        #     prediction = p_sent.max(1)[1]  # [node]
        #     pred_idx = torch.arange(N)[prediction != 0].long()
        # else:
        #     if blocking:
        #         pred_idx = self.ngram_blocking(original_article_sents, p_sent[:, 1], self.blocking_win,
        #                                        min(m, N))
        #     else:
        # topk, pred_idx = torch.topk(p_sent.squeeze(), min(m, N))
        topk, pred_idx = torch.topk(p_sent[:, 1], min(self.m, N))
        # thr = 0.45
        # pred_idx = torch.where(p_sent.squeeze() > thr)
        # if len(pred_idx) == 0:

        return topk, pred_idx

    def tensor_similarity(self, t1, t2):
        combine_counts = torch.cat((t1, t2)).unique(return_counts=True)[1]
        return (combine_counts > 1).sum() / t2.unique(return_counts=True)[1].shape[0]

    def select_sentences(self, graph, m=4):
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        N = len(snode_id)
        p_sent = graph.ndata["p"][snode_id]
        words = graph.nodes[snode_id].data["words"]
        sorted_idx = torch.sort(p_sent.squeeze(), descending=True)[1]
        select_idx = [sorted_idx[0].item()]
        selected_words = torch.Tensor().to(self.hps.device)
        i = 1
        while len(select_idx) < m and i < N:
            _words = words[sorted_idx[i]]
            s = self.tensor_similarity(selected_words, _words)
            if s.item() < 0.8:
                select_idx.append(sorted_idx[i].item())
                selected_words = torch.cat([selected_words, _words])
            i += 1

        return select_idx

    def evaluation(self, G, index, dataset, blocking=False):
        """
            :param G: the model
            :param index: list, example id
            :param dataset: dataset which includes text and summary
            :param blocking: bool, for n-gram blocking
        """
        self.batch_number += 1
        outputs = self.model.forward(G)
        # outputs, word_outs = self.model.forward(G)
        # word_node_id = G.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        # G.nodes[word_node_id].data["word_out"] = word_outs

        # logger.debug(outputs)
        snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        label = G.ndata["label"][snode_id].sum(-1)  # [n_nodes]
        G.nodes[snode_id].data["loss"] = self.criterion(outputs, label).unsqueeze(-1)  # [n_nodes, 1]
        # scores = label
        # scores = self.get_scores(G, index)
        # G.nodes[snode_id].data["loss"] = self.criterion(outputs.squeeze(), scores.to(self.hps.device)).unsqueeze(
        #     -1)  # [n_nodes, 1]
        loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
        # loss = dgl.mean_nodes(G, "loss")  # [batch_size, 1]
        loss = loss.mean()
        self.running_loss += float(loss.data)

        # G.nodes[snode_id].data["p"] = scores
        G.nodes[snode_id].data["p"] = outputs
        glist = dgl.unbatch(G)
        for j in range(len(glist)):
            g = glist[j]

            idx = index[j]
            example = dataset.get_example(idx)
            original_article_sents = example.original_article_sents
            sent_max_number = len(original_article_sents)
            refer = example.original_abstract

            snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            N = len(snode_id)
            p_sent = g.ndata["p"][snode_id]
            self.outs.append(p_sent.squeeze().tolist())
            # p_sent = p_sent.view(-1, 2)  # [node, 2]
            label = g.ndata["label"][snode_id].sum(-1).squeeze().cpu()  # [n_node]
            topk, pred_idx = self.get_m_top(p_sent=p_sent, N=N, m=self.hps.m)
            # pred_idx = p_sent.argwhere(p_sent >= 0.5)

            # pred_idx = self.select_sentences(graph=g, m=self.hps.m)

            # topk, pred_idx = self.get_pred_idx__(g)
            # if len(pred_idx)==0:
            #     topk, pred_idx = self.get_m_top(p_sent=p_sent, N=N, m=1)

            prediction = torch.zeros(N).long()
            prediction[pred_idx] = 1
            self.extracts.append(pred_idx)

            self.pred += prediction.sum()
            self.true += label.sum()

            self.match_true += ((prediction == label) & (prediction == 1)).sum()
            self.match += (prediction == label).sum()
            self.total_sentence_num += N
            self.example_num += 1
            hyps = "\n".join(original_article_sents[id] for id in pred_idx if id < sent_max_number)

            self._hyps.append(hyps)
            self._refer.append(refer)

    def getMetric(self):
        logger.info("[INFO] Validset match_true %d, pred %d, true %d, total %d, match %d",
                    self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        self._accu, self._precision, self._recall, self._F = eval_label(
            self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        logger.info(
            "[INFO] The size of totalset is %d, sent_number is %d, accu is %f, precision is %f, recall is %f, F is %f",
            self.example_num, self.total_sentence_num, self._accu, self._precision, self._recall, self._F)

    def ngram_blocking(self, sents, p_sent, n_win, k):
        """
        
        :param p_sent: [sent_num, 1]
        :param n_win: int, n_win=2,3,4...
        :return:
        """
        ngram_list = []
        _, sorted_idx = p_sent.sort(descending=True)
        S = []
        for idx in sorted_idx:
            sent = sents[idx]
            pieces = sent.split()
            overlap_flag = 0
            sent_ngram = []
            for i in range(len(pieces) - n_win):
                ngram = " ".join(pieces[i: (i + n_win)])
                if ngram in ngram_list:
                    overlap_flag = 1
                    break
                else:
                    sent_ngram.append(ngram)
            if overlap_flag == 0:
                S.append(idx)
                ngram_list.extend(sent_ngram)
                if len(S) >= k:
                    break
        S = torch.LongTensor(S)
        # print(sorted_idx, S)
        return S

    @property
    def labelMetric(self):
        return self._F
