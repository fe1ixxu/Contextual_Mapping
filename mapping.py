import torch
import numpy as np
from src.trainer import trainer
from src.evaluator import evaluator
from src.utility import *
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tgt", type=str, default = "de")
    parser.add_argument("--emb_path", type=str, default="en-de_word_vectors/aligned/")
    parser.add_argument("--dict_path", type=str, default = "./data/dict/")
    parser.add_argument("--vocab_path", type=str, default = "./data/vocab/")
    parser.add_argument("--random_num_d", type=int, default = 1000) # number of randomly selected vectors for isotropy and isometry 
    parser.add_argument("--random_num_rs", type=int, default = 1500) # number of words (senses) for RS
    parser.add_argument("--csls_k", type=int, default = 10)
    parser.add_argument("--if_iter_norm", type=bool, default = True)


    args = parser.parse_args()

    src = 'en'
    tgt = args.tgt

    train_file = args.dict_path + src + "-" + tgt + ".0-5000.txt"
    test_file = args.dict_path + src + "-" + tgt + ".5000-6500.txt"
    
    with open(args.vocab_path + src + "_200k.txt", encoding="utf-8") as f:
        src_constraint = f.readlines()
        src_constraint = [s.strip() for s in src_constraint]

    with open(args.vocab_path + tgt + "_200k.txt", encoding="utf-8") as f:
        tgt_constraint = f.readlines()
        tgt_constraint = [t.strip() for t in tgt_constraint]


    print("mono-sense ---------------------------------")
    train_mean_np = trainer(args.emb_path + "mean_"+src+"-"+tgt, train_file, src_constraint, tgt_constraint, args.if_iter_norm)
    evaluate = evaluator(train_mean_np, test_file)
    W, aligned = train_mean_np.simple_procrustes()
    evaluate.calculate_accuracy(aligned, csls_k=args.csls_k)
    index = random_indice_generator(args.random_num_d, train_mean_np.source_vector[:,train_mean_np.index].shape[1])

    print("anisotriopy: ", 
        degree_anisotropy(train_mean_np.source_vector[:,train_mean_np.index], index),
        degree_anisotropy(train_mean_np.target_vector[:,train_mean_np.index], index))
    print("anisometry: ", degree_isometric(train_mean_np.source_vector[:,train_mean_np.index], 
        train_mean_np.target_vector[:,train_mean_np.index], index))
    print("rs:", cal_rs(train_mean_np.source_vector[:, train_mean_np.index], 
        train_mean_np.target_vector[:, train_mean_np.index], args.random_num_rs))

    
    print("multi-sense ---------------------------------")
    train_multi_np = trainer(args.emb_path + "/multi_"+src+"-"+tgt, train_file, 
        set(train_mean_np.source), set(train_mean_np.target), args.if_iter_norm)
    evaluate = evaluator(train_multi_np, test_file)
    W, aligned = train_multi_np.simple_procrustes()
    evaluate.calculate_accuracy(aligned, csls_k=args.csls_k)
    index = random_indice_generator(args.random_num_d, train_multi_np.source_vector[:, train_multi_np.index].shape[1])

    print("anisotriopy: ", degree_anisotropy(train_multi_np.source_vector[:,train_multi_np.index], index),
        degree_anisotropy(train_multi_np.target_vector[:,train_multi_np.index], index))
    print("anisometry: ", degree_isometric(train_multi_np.source_vector[:,train_multi_np.index], 
        train_multi_np.target_vector[:,train_multi_np.index], index))
    print("rs:", cal_rs(train_multi_np.source_vector[:, train_multi_np.index],
                 train_multi_np.target_vector[:, train_multi_np.index], args.random_num_rs))
    

if __name__ == '__main__':
    main()

