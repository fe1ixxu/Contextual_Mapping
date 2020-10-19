# Cross-Lingual Contextual Embedding Space Mapping
![sense-level](https://github.com/fe1ixxu/Contextual_Mapping/blob/main/figures/sense-level.png)
Example of sense-level mapping: 'bank' is split into two sense embeddings, which are respectively mapped to German 'bank' (financial establishment) and 'ufer' (shore). Similar discussion also applies to the word 'hard', where its two sense vectors are mapped to German 'schwer' (difficult) and 'hart' (solid).

## Prerequisites
First install the virtual environmemt including required packages.

```
conda create --name cmap python=3.7
conda activate cmap
pip install -r requirements.txt
```

## Use Pre-Trained Embeddings and Reproduce the Number in the Paper
|    Embeddings    | 
| ---------- | 
| [En-De (word-level)](https://drive.google.com/file/d/1Yg6hkRvVbF0by34JA02uiJl-4BidI-nN/view?usp=sharing) | 
| [En-De (sense-level)](https://drive.google.com/file/d/1dFb0lxqlBZpjLTmLYiKdxE-Kh48T2SBc/view?usp=sharing) |
| [En-Ar (word-level)](https://drive.google.com/file/d/16Dj3I61sMWqjdXrbWerKFEHquCbV8ZoR/view?usp=sharing) |
| [En-Ar (sense-level)](https://drive.google.com/file/d/1P91Yw2CTkT4TTW99O3dasqq95a73zyjz/view?usp=sharing) |
| [En-Nl (word-level)](https://drive.google.com/file/d/1y0bLLasdsKRxlnadMk5qt8O8T4vjzESn/view?usp=sharing)|
| [En-Nl (sense-level)](https://drive.google.com/file/d/17ojesGWxFMQfo9v19os2FJE4JHeoM0sh/view?usp=sharing)| 

## Bilingual Dictionary Induction
An example of evaluating English-German mapping through BDI. The results of isotropy, isometry and isomorphism will also print out.
```
python mapping --tgt de --emb_path $EMB/PATH --if_iter_norm True
```

## To Create Your Own Aligned Embeddings:
Alternatively, if you want to get your own customized aligned embeddings, please see the following instrctions.

### 1. Installing Fast Align

Please install the [fast align](https://github.com/clab/fast_align) toolkit by following their instruction.

### 2. Download and Preprocess Parallel Corpora

Parallel Corpora are downloaded from [ParaCrawl](https://www.paracrawl.eu/). A preprocessed file is already prepared for you and you can easily run it by one command. An example of downloading and preprocessing En-De parallel corpora:
```
wget https://s3.amazonaws.com/web-language-models/paracrawl/release5.1/en-de.txt.gz
gunzip ./en-de.txt.gz
./data/preprocess.sh de YOUR/PATH/FOR/PARALLEL/CORPUS YOUR/PATH/FOR/FAST/ALIGN
```

As for the En-Ar parallel corpus, please find all preprocessed file [here](https://drive.google.com/drive/folders/1bHD7aATC0ZVio-c3kqbiFzoZzD9DN2qt?usp=sharing).

### 3. Obtain Contextual Embeddings

Continuing the above example, we run `getwordvectorsfrombert.py` to obtain aligned contextual embeddings.
```
path=YOUR/PATH/FOR/PARALLEL/CORPUS
python getwordvectorsfrombert.py --src en --tgt de --open_src_file ${path}en_token.txt  --open_tgt_file ${path}de_token.txt  --open_align_file ${path}forward_align.txt --write_vectors_path ${path}vectors/ --max_num_word 10000 --batch_size 256 --max_seq_length 150
```

![vectors](https://github.com/fe1ixxu/Contextual_Mapping/blob/main/figures/vectors.jpg)

### 4. Cluster aligned Embeddings

To obtain sense-level embeddings:
```
python cluster_vector.py --input_file ${path}vectors/ --write_file $output --stopwords ./data/stopwords/en.txt --min_threshold 100 --min_num_words 5 
```
To obtain word-level embeddings, we just simply increase the threshold of clustering (vectors will be clusterd if its occurance is higher than the thereshold) to a large number (>10000).

```
python cluster_vector.py --input_file ${path}vectors/ --write_file $output --stopwords ./data/stopwords/en.txt --min_threshold 100000 --min_num_words 5 
```

The output contextual aligned embeddings file includes 5 columns. They respectively represents:
```
0 - source word (sense)
1 - translated word (sense) in the target side
2 - occurance of the source word
3 - entropy of the cluster (didn't use for this paper)
4 - source word (sense) embedding
5 - target word (sense) embedding
```



