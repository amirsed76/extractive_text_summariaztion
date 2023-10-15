import json
import os
import shutil
import time
import numpy as np
import torch
import dgl
from config import _DEBUG_FLAG_
from data_manager import data_loaders
from tools.logger import *
from tools.utils import save_model
from runner.evaluation import run_eval
import pandas as pd
from utils import CustomLoss, RegressionCustomLoss


def setup_training(model, hps, data_variables):
    train_dir = os.path.join(hps.save_root, "train")

    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        # best_model_file = os.path.join(train_dir, hps.restore_model)
        # model.load_state_dict(torch.load(best_model_file))
        model.load_state_dict(torch.load(hps.restore_model))
        # model.load_state_dict(torch.load(hps.restore_model))
        # hps.save_root = hps.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, hps, data_variables=data_variables)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


class Trainer:
    def __init__(self, model, hps, train_dir):
        self.model = model
        self.hps = hps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hps.lr)
        #
        # self.optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.model.parameters()), lr=hps.lr)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.Tensor([0.1, 0.9]).to(hps.device))
        # self.criterion = CustomLoss()
        # self.word_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.best_train_loss = None
        self.best_loss = None
        self.best_F = None
        self.non_descent_cnt = 0
        self.saveNo = 0
        self.epoch = 1
        self.epoch_avg_loss = 0
        self.train_dir = train_dir
        self.report_epoch = 2
        # self.report_epoch = 120
        self.scores = pd.read_csv(f"{os.path.join(hps.data_dir, 'train.csv')}")["scores"].values.tolist()

    def run_epoch(self, train_loader):
        epoch_start_time = time.time()

        train_sentence_loss = 0.0
        # train_word_loss = 0.0
        epoch_loss = 0.0
        iters_start_time = time.time()
        # iter_start_time = time.time()
        for i, (G, index) in enumerate(train_loader):
            sentence_loss = self.train_batch(G=G, index=index)
            # print(f"{i}=>{loss}")

            train_sentence_loss += float(sentence_loss.data)
            # train_word_loss += float(word_loss.data)
            epoch_loss += float(sentence_loss.data)
            if i % self.report_epoch == self.report_epoch - 1:
                if _DEBUG_FLAG_:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())
                batch_time_sum = time.time() - iters_start_time
                iters_start_time = time.time()

                logger.info(
                    '| end of iter {:3d} | time: {:5.2f}s | sentence loss {:5.4f}'.format(i, (
                            batch_time_sum / self.report_epoch), float(train_sentence_loss / self.report_epoch)))

                train_sentence_loss = 0.0
                # train_word_loss = 0.0
                self.save_current_model()
            # iter_start_time = time.time()

        # self.save_epoch_model()

        self.epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info(' | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '.format(self.epoch, (
                time.time() - epoch_start_time), float(self.epoch_avg_loss)))
        return epoch_loss

    def get_scores(self, G, indexes):
        scores = torch.Tensor()
        for g, index in zip(dgl.unbatch(G), indexes):
            snod_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)

            scores = torch.cat([scores, torch.Tensor(json.loads(self.scores[index])[:len(snod_id)])])

        return scores.to(self.hps.device)

    def get_important_words(self, G):
        snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        labels = G.ndata["label"][snode_id].sum(-1)
        summary_node_id = snode_id[labels.argwhere().squeeze()]
        summary_words = G.nodes[summary_node_id].data["words"].unique()
        # un_summary_node_id = np.setdiff1d(snode_id.cpu().numpy(), summary_node_id.cpu().numpy())
        # un_summary_words = G.nodes[un_summary_node_id].data["words"].unique()
        # important_word_ids = np.setdiff1d(summary_words.cpu().numpy(), un_summary_words.cpu().numpy())
        return summary_words
        # return torch.from_numpy(important_word_ids).to(self.hps.device)

    def get_labels(self, G):
        graph_list = dgl.unbatch(G)
        labels = torch.Tensor().to(self.hps.device)
        for graph in graph_list:
            word_node_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)

            word_ids = graph.nodes[word_node_id].data["id"]
            summary_word_ids = self.get_important_words(graph)
            # labels += [1 if _id in summary_word_ids else 0 for _id in word_ids.tolist()]
            labels = torch.cat([labels, torch.isin(word_ids, summary_word_ids).int()])
        return torch.Tensor(labels).to(self.hps.device).long()

    def train_batch(self, G, index):
        G = G.to(self.hps.device)
        outputs = self.model.forward(G)  # [n_snodes, 2]
        # outputs, word_out = self.model.forward(G)  # [n_snodes, 2]
        snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        # scores = self.get_scores(G, index)
        # word_node_id = G.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        #
        # word_label = self.get_labels(G)
        # word_label = G.ndata["label"][word_node_id]
        label = G.ndata["label"][snode_id].sum(-1)  # [n_nodes]
        # scores = label.float()
        # sentence_losses = self.criterion(outputs.squeeze(),
        #                                  scores.to(self.hps.device)).unsqueeze(-1)  # [n_nodes, 1]
        sentence_losses = self.criterion(outputs,
                                         label.to(self.hps.device)).unsqueeze(-1)  # [n_nodes, 1]

        # word_losses = self.word_criterion(word_out, word_label)

        G.nodes[snode_id].data["loss"] = sentence_losses
        # G.nodes[word_node_id].data["word_loss"] = word_losses

        # G.nodes[snode_id].data["loss"] = self.criterion(outputs,
        #                                                 label.to(self.hps.device)).unsqueeze(-1)  # [n_nodes, 1]

        sentence_loss = dgl.sum_nodes(G, "loss").mean()
        # word_loss = dgl.sum_nodes(G, "word_loss").mean()

        # sentence_loss = sentence_losses.mean()
        # loss = dgl.mean_nodes(G, "loss")  # [batch_size, 1]
        # loss = 9 * sentence_loss + word_loss
        loss = sentence_loss
        if not (np.isfinite(loss.data.cpu())).numpy():
            logger.error("train Loss is not finite. Stopping.")
            logger.info(loss)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.info(name)
                    # logger.info(param.grad.data.sum())
            raise Exception("train Loss is not finite. Stopping.")
        self.optimizer.zero_grad()
        loss.backward()
        if self.hps.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hps.max_grad_norm)

        self.optimizer.step()
        # return torch.mean(sentence_losses), torch.mean(word_losses)
        return loss
        # return torch.mean(sentence_losses)

    def change_learning_rate(self):
        if self.hps.lr_descent:
            new_lr = max(5e-6, self.hps.lr / (self.epoch + 1))
            for param_group in list(self.optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

    def save_epoch_model(self):
        if not self.best_train_loss or self.epoch_avg_loss < self.best_train_loss:
            save_file = os.path.join(self.train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s',
                        float(self.epoch_avg_loss),
                        save_file)
            save_model(self.model, save_file)
            self.best_train_loss = self.epoch_avg_loss
        elif self.epoch_avg_loss >= self.best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(self.model, os.path.join(self.train_dir, "earlystop"))
            sys.exit(1)

    def save_current_model(self):
        save_file = os.path.join(self.train_dir, "current")
        save_model(self.model, save_file)


def run_training(model, hps, data_variables):
    trainer = Trainer(model=model, hps=hps, train_dir=os.path.join(hps.save_root, "train"))
    train_size = 287000
    n_part = 80

    print(f"data_loader")
    logger.info(model)

    for epoch in range(1, hps.n_epochs + 1):
        # for epoch in range(1, 1 + 1):
        logger.info(f"train started in epoch={epoch}")

        logger.info("train loader read")

        trainer.epoch = epoch
        model.train()

        for train_data_part in range(n_part + 1):
            # for train_data_part in range(1):
            if train_data_part == n_part:
                from_index = train_data_part * train_size // n_part
                to_index = None
            else:
                from_index = train_data_part * train_size // n_part
                to_index = (train_data_part + 1) * train_size // n_part
            train_loader = data_loaders.make_dataloader(data_file=data_variables["train_file"],
                                                        vocab=hps.vocab, hps=hps,
                                                        filter_word=data_variables["filter_word"],
                                                        w2s_path=data_variables["train_w2s_path"],
                                                        graphs_dir=os.path.join(data_variables["graphs_dir"],
                                                                                "train"),
                                                        from_index=from_index,
                                                        to_index=to_index,
                                                        shuffle=True
                                                        )

            print(f"train loader from {from_index} to {to_index} started epoch started ")

            trainer.run_epoch(train_loader=train_loader)
            print(f"train loader from {from_index} to {to_index} started epoch finished ")
            del train_loader

        valid_loader = data_loaders.make_dataloader(data_file=data_variables["valid_file"],
                                                    vocab=hps.vocab, hps=hps,
                                                    filter_word=data_variables["filter_word"],
                                                    w2s_path=data_variables["val_w2s_path"],
                                                    graphs_dir=os.path.join(data_variables["graphs_dir"],
                                                                            "test"))

        # valid_loader = data_loaders.make_dataloader(data_file=data_variables["valid_file"],
        #                                             vocab=data_variables["vocab"], hps=hps,
        #                                             filter_word=data_variables["filter_word"],
        #                                             w2s_path=data_variables["val_w2s_path"],
        #                                             graphs_dir=os.path.join(data_variables["graphs_dir"],
        #                                                                     "val"))

        best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valid_loader.dataset, hps,
                                                              trainer.best_loss,
                                                              trainer.best_F, trainer.non_descent_cnt,
                                                              trainer.saveNo)
        trainer.best_F = best_F
        trainer.best_loss = best_loss
        trainer.non_descent_cnt = non_descent_cnt
        trainer.saveNo = saveNo
        save_model(model, os.path.join(hps.save_root, f"epoch_{epoch}"))

        del valid_loader

        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(hps.save_root, "earlystop"))
            return
