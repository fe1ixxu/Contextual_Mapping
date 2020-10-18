#!/bin/bash

src=$en
tgt=$1
path=/export/b15/haoranxu/word_vectors/$src-$tgt-word_vectors/

mkdir ${path}vectors/
mkdir ${path}aligned/

python create_sen.py --file $path$src-$tgt.txt --en_file $path$src.txt --lg_file $path$tgt.txt
python bert_token_sen.py --src $src --tgt $tgt --src_file $path${src}.txt --tgt_file $path${tgt}.txt --write_src_file $path${src}_token.txt --write_tgt_file $path${tgt}_token.txt
python para-src-tgt.py --src_file $path${src}_token.txt --tgt_file $path${tgt}_token.txt --write_file ${path}para_${src}_${tgt}.txt
../fastalign/fast_align-master/build/fast_align -i ${path}para_${src}_${tgt}.txt -d -o -v >${path}forward_align.txt

rm ${path}${src}.txt
rm ${path}${tgt}.txt
rm ${path}para_${src}_${tgt}.txt
