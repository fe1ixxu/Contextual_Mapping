import torch

class evaluator():
    def __init__(self, trainer, file):
        self.trainer = trainer
        self.test_dict = None
        self.index = []
        self.reading_test_dictionary(file)


    def reading_test_dictionary(self, file):
        assert self.index == []
        gold_dict = {}
        with open(file, encoding="utf-8") as f:
            line = f.readline()
            while (line):
                line = line.split()
                if line[0] in self.trainer.source and line[1] in self.trainer.target:
                    self.index.append(self.trainer.source.index(line[0]))  #find the first word whose occurance is highest
                    if line[0] not in gold_dict:
                        gold_dict[line[0]] = [line[1]]
                    else:
                        gold_dict[line[0]].append(line[1])
                line = f.readline()
        self.test_dict = gold_dict
        self.index = list(set(self.index))
        print("We are considering {} words in dictionary to test the accuracy".format(len(self.index)))

    def find_all_souce(self, word):
        index = []
        for ind, src in enumerate(self.trainer.source):
            if src == word:
                index.append(ind)
        return index


    def calculate_accuracy(self, aligned, csls_k=10):
        acc_nn = 0
        acc_nn_5 = 0
        acc_csls = 0
        acc_csls_5 = 0
        total = 0

        for ind in self.index:
            source_vec = aligned[:, ind]
            words = self.get_knn_words(source_vec)
            if len(set(words[:1]+self.test_dict[self.trainer.source[ind]])) < len(words[:1]) + len(self.test_dict[self.trainer.source[ind]]):
                acc_nn += 1
            if len(set(words[:5]+self.test_dict[self.trainer.source[ind]])) < len(words[:5]) + len(self.test_dict[self.trainer.source[ind]]):
                acc_nn_5 += 1

            total += 1
        print("The accuracy of {}nn is {}%".format(1, acc_nn * 100 / total))
        print("The accuracy of {}nn is {}%".format(5, acc_nn_5 * 100 / total))


        if not csls_k:
            return

        for ind in self.index:
            source_vec = aligned[:, ind]
            words = self.get_csls_words(source_vec, aligned, csls_k)
            if len(set(words[:1]+self.test_dict[self.trainer.source[ind]])) < len(words[:1]) + len(self.test_dict[self.trainer.source[ind]]):
                acc_csls += 1
            if len(set(words[:5]+self.test_dict[self.trainer.source[ind]])) < len(words[:5]) + len(self.test_dict[self.trainer.source[ind]]):
                acc_csls_5 += 1

        print("The accuracy P@{} csls is {}%".format(1 ,acc_csls * 100 / total))
        print("The accuracy P@{} csls is {}%".format(5, acc_csls_5 * 100 / total))


    def get_knn_words(self, vec):
        cos_sim = self.trainer.target_vector.t() @ vec.view(-1,1)
        _, ind = torch.sort(cos_sim.view(-1), descending=True)
        words = [self.trainer.target[i] for i in ind[:10]]
        return words

    def get_csls_words(self, vec, aligned, csls_k):
        cos_sim = self.trainer.target_vector.t() @ vec.view(-1,1)
        cos_sim, index = torch.sort(cos_sim.view(-1), descending=True)
        index = index[:20]
        r_s = cos_sim[:csls_k].mean(0)

        r_t = aligned.t() @ self.trainer.target_vector[:,index]
        r_t, _ = torch.sort(r_t, dim=0, descending=True)
        r_t = r_t[: csls_k].mean(dim=0).view(-1)

        dis = 2*cos_sim[:10] - r_s - r_t[:10]
        dis = list(zip(dis, index))
        dis.sort(key=lambda x:x[0], reverse=True)
        dis = list(map(lambda x:x[1], dis))
        words = [self.trainer.target[i] for i in dis[:10]]
        return words
