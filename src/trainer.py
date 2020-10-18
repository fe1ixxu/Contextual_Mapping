import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, AutoTokenizer
import numpy as np

class trainer():
    def __init__(self, input_file, train_file, src_cons, tgt_cons, ifiter):
        self.train_file = train_file
        self.ifiter = ifiter
        self.source = []
        self.target = []
        self.source_occur = []
        self.source_vector = []
        self.target_vector = []
        self.info = None
        self.index = None
        self.s_train = None
        self.t_train = None
        self.src_cons = src_cons
        self.tgt_cons = tgt_cons

        self.source_index = 0
        self.target_index = 1
        self.occur_index = 2
        self.entropy_index = 3
        self.source_vector_index = 4
        self.target_vector_index = 5

        self.reading_file(input_file)
        self.load_trainning_data()

    def reading_file(self, input_file):
        with open(input_file, encoding="utf-8") as f:
            line = f.readline()
            while(line):
                line = line.split("\t")
                if line[self.source_index] in self.src_cons and line[self.target_index] in self.tgt_cons:
                    self.source.append(line[self.source_index])
                    self.target.append(line[self.target_index])
                    self.source_occur.append(int(line[self.occur_index]))
                    vector_en = np.array(line[self.source_vector_index].split()[:768], dtype=float)
                    self.source_vector.append(torch.tensor(vector_en, dtype=torch.float))
                    vector_de = np.array(line[self.target_vector_index].split()[:768], dtype=float)
                    self.target_vector.append(torch.tensor(vector_de, dtype=torch.float))
                line = f.readline()

        self.get_sort()

        self.source_vector = torch.stack(self.source_vector)
        self.source_vector /= torch.norm(self.source_vector, dim=1).view(-1,1)
        self.source_vector = self.source_vector.t()
        self.target_vector = torch.stack(self.target_vector)
        self.target_vector /= torch.norm(self.target_vector, dim=1).view(-1, 1)
        self.target_vector = self.target_vector.t()

        self.info = list(zip(self.source,self.target, range(len(self.source))))

        if self.ifiter:
            self.iter_norm()

    def load_trainning_data(self):
        self.s_train = set()
        self.t_train = set()
        print("Loading training data...")
        with open(self.train_file, encoding="utf-8") as f:
            line = f.readline()
            while(line):
                line = line.split()
                s = line[0]
                t = " ".join(line[1:])
                if s in self.source and t in self.target: 
                    self.s_train.add(s)
                    self.t_train.add(t)
                line = f.readline()
        print("Training data loaded.")
        self.index = list(
            filter(lambda x: x[0] in self.s_train and x[1] in self.t_train, self.info))
        self.index = list(map(lambda x: x[-1], self.index))

    def simple_procrustes(self):
        source = self.source_vector[:, self.index]
        target = self.target_vector[:, self.index]
        U, _, V = torch.svd(target @ source.t())
        W= U @ V.t()
        aligned = W @ self.source_vector
        return W, aligned

    def close_pair_procrustes(self, aligned):
        aligned = aligned[:,self.index]
        cos_sim1 = self.target_vector[:,self.index].t() @ aligned
        cos_sim2 = aligned.t() @ self.target_vector[:,self.index]
        cos_sim1, ind_1 = torch.sort(cos_sim1, dim=0, descending=True)
        cos_sim2, ind_2 = torch.sort(cos_sim2, dim=0, descending=True)

        source = []
        target = []
        for i in range(ind_1.shape[0]):
            if ind_2[0, ind_1[0,i]] == i:
                source.append(self.source_vector[:,self.index[i]])
                target.append(self.target_vector[:,self.index[i]])
        print(len(source))
        source = torch.stack(source).t()
        target = torch.stack(target).t()
        U, _, V = torch.svd(target @ source.t())
        W = U @ V.t()
        aligned = W @ self.source_vector
        return W, aligned


    def get_sort(self):
        combine = list(zip(self.source, self.target, self.source_vector, self.target_vector, self.source_occur))
        combine.sort(key = lambda x: x[-1], reverse=True)
        self.source = list(map(lambda x:x[0], combine))
        self.target = list(map(lambda x:x[1], combine))
        self.source_vector = list(map(lambda x:x[2], combine))
        self.target_vector = list(map(lambda x:x[3], combine))
        self.source_occur = list(map(lambda x:x[-1], combine))

    def iter_norm(self, num=5):
        for _ in range(num):
            self.source_vector -= self.source_vector.mean(dim=1).view(-1, 1)
            self.target_vector -= self.target_vector.mean(dim=1).view(-1, 1)
            self.source_vector /= torch.norm(self.source_vector, dim=0).view(1,-1)
            self.target_vector /= torch.norm(self.target_vector, dim=0).view(1, -1)
