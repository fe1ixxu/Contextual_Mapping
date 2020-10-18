import torch
import random

def cal_rs(aligned, targets, k):
    # index = torch.randint(0, aligned.shape[1], (k,))
    list_s = aligned[:,:k].t() @ aligned[:,:k]
    list_t = targets[:,:k].t() @ targets[:,:k]

    filter_st = torch.triu(-100*torch.ones(list_s.shape[1], list_s.shape[1])).t()
    list_s = torch.where(filter_st == -100, filter_st, list_s)
    list_t = torch.where(filter_st == -100, filter_st, list_t)

    ind_s = (list_s != -100).nonzero()
    ind_t = (list_t != -100).nonzero()

    list_s = list_s[ind_s[:,0], ind_s[:,1]]
    list_t = list_t[ind_t[:,0], ind_t[:,1]]

    list_s = list_s - torch.mean(list_s)
    list_t = list_t - torch.mean(list_t)

    p = torch.sum(list_s*list_t)/(torch.sqrt(torch.sum(list_s*list_s))*torch.sqrt(torch.sum(list_t*list_t)))
    return p


def degree_anisotropy(vectors, index):
    vectors = torch.triu(vectors[:, index].t() @ vectors[:, index], 1)
    filter_st = torch.triu(-100*torch.ones(vectors.shape[1], vectors.shape[1])).t()
    vectors = torch.where(filter_st == -100, filter_st, vectors)
    ind = (vectors != -100).nonzero()
    vectors = vectors[ind[:, 0], ind[:, 1]]

    return torch.mean(vectors)

def degree_isometric(vec_s, vec_t, index):

    vec_s = torch.triu(vec_s[:, index].t() @ vec_s[:, index], 1)
    vec_t = torch.triu(vec_t[:, index].t() @ vec_t[:, index], 1)

    filter_st = torch.triu(-100*torch.ones(vec_s.shape[1], vec_s.shape[1])).t()
    vec_s = torch.where(filter_st == -100, filter_st, vec_s)
    ind = (vec_s != -100).nonzero()

    vec_s = vec_s[ind[:,0], ind[:,1]]
    vec_t = vec_t[ind[:,0], ind[:,1]]

    diff = 2 * torch.abs(vec_s - vec_t)

    return torch.mean(diff)

def random_indice_generator(num, maxnum):
    index = list(range(maxnum))
    index = random.sample(index, num)
    index = torch.tensor(index, dtype=torch.long)

    return index

