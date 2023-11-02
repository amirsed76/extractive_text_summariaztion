import os

import dgl
import torch.utils.data
from dgl.data.utils import load_graphs
import spacy
from data_manager.dataloader import Example, read_json

nlp = spacy.load("en_core_web_sm")


class CachedSummarizationDataSet(torch.utils.data.Dataset):

    def __init__(self, hps, data_type="train", data_path=None, vocab=None, from_index=0, to_index=0):
        self.hps = hps
        self.sent_max_len = hps.sent_max_len
        self.doc_max_timesteps = hps.doc_max_timesteps
        self.max_instance = hps.max_instances
        self.graphs_dir = os.path.join(self.hps.graphs_dir, data_type)
        self.syntax_graphs_dir = os.path.join(self.hps.syntax_graphs_dir, data_type)
        self.use_cache = self.hps.fill_graph_cache
        self.from_index = from_index
        self.to_index = to_index
        self.graph_index_from = 0
        self.graph_index_offset = 256
        root, _, files = list(os.walk(self.graphs_dir))[0]
        indexes = [int(item[:-4]) for item in files]
        max_index = max(indexes)
        # size = max(indexes) + self.graph_index_offset - 1
        g, label_dict = load_graphs(os.path.join(root, f"{max_index}.bin"))
        size = max_index + len(g) - 1
        if to_index is None:
            to_index = size
        max_instances = hps.max_instances if hps.max_instances else 288000
        self.size = min([to_index - from_index, max_instances, size])
        self.graphs = dict()
        self.syntax_graphs = dict()
        self.load_HSG_graphs()
        self.example_list = None
        self.vocab = vocab
        self.data_path = data_path
        self.fill_example_list()

    def fill_example_list(self):
        self.example_list = read_json(self.data_path, max_instance=self.max_instance,
                                      from_instances_index=self.hps.from_instances_index)

    def get_example(self, index):
        if self.example_list is None:
            self.fill_example_list()

        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def load_HSG_graphs(self):
        graphs, _ = load_graphs(os.path.join(self.graphs_dir, f"{self.graph_index_from}.bin"))
        for i, graph in enumerate(graphs):
            self.graphs[self.graph_index_from + i] = graph

    def load_syntax_graphs(self):
        graphs, _ = load_graphs(os.path.join(self.syntax_graphs_dir, f"{self.graph_index_from}.bin"))
        for i, graph in enumerate(graphs):
            self.syntax_graphs[self.graph_index_from + i] = graph

    def get_graph(self, index):
        if index not in self.graphs.keys():
            self.graph_index_from = (index // self.graph_index_offset) * self.graph_index_offset
            self.load_HSG_graphs()

        return self.graphs[index]

    def get_syntax_graph(self, index):
        if index not in self.syntax_graphs.keys():
            self.graph_index_from = (index // self.graph_index_offset) * self.graph_index_offset
            self.load_syntax_graphs()

        return self.syntax_graphs[index]

    def __getitem__(self, index):

        try:
            G = self.get_graph(index)
            syntax_graph = self.get_syntax_graph(index=index)
            return G, syntax_graph, index
        except Exception as e:
            raise e
            print(f"EXCEPTION => {e}")
            return None

    def __getitems__(self, possibly_batched_index):
        result = []
        for index in possibly_batched_index:
            item = self.__getitem__(self.from_index + index)
            if item is not None:
                result.append(item)

        return result

    def __len__(self):
        return self.size
