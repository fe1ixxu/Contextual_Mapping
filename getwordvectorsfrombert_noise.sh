#!/bin/bash
source /home/haoranxu/Anaconda/python3/bin/activate allennlp

src=en
tgt=de
MAX_WORD=10000 # man num of words to store

path=/export/c12/haoranxu/test_vectors/
CUDA_VISIBLE_DEVICES=`free-gpu` python getwordvectorsfrombert.py --src $src --tgt $tgt --open_src_file ${path}${src}_token.txt  --open_tgt_file ${path}${tgt}_token.txt  --open_align_file ${path}forward_align.txt --write_vectors_path ${path}vectors/ --max_num_word $MAX_WORD --batch_size 256 --max_seq_length 150



ifcluster=yes
if [ ${ifcluster} == yes ]
then
    output=${path}aligned/multi_${src}-${tgt}
    max=5
else
    output=${path}aligned/mean_${src}-${tgt}
    max=100000
fi

python cluster_vector.py --input_file ${path}vectors/ --write_file $output --stopwords ./data/stopwords/${src}.txt --min_threshold $max --min_num_words 5 --start 0 --end -1 

ifcluster=no
if [ ${ifcluster} == yes ]
then
    output=${path}aligned/multi_${src}-${tgt}
    max=5
else
    output=${path}aligned/mean_${src}-${tgt}
    max=100000
fi

python cluster_vector.py --input_file ${path}vectors/ --write_file $output  --stopwords ./data/stopwords/${src}.txt --min_threshold $max --min_num_words 5 --start 0 --end -1 