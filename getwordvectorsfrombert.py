from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""This script was copied and modified from https://github.com/huggingface/pytorch-pretrained-BERT"""
"""Note: this version of the code doesn't handle words that are split into different subwords"""
###
#path in code should be refined and more robust for users
###


from tqdm import tqdm
import argparse
import numpy as np
import collections
import logging
import fcntl
import json
import re
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel, AutoConfig

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)

        tokens += tokens_a
        input_type_ids += [0]*len(tokens_a)

        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def get_tokenizer_model(language):
    assert language in ["en", "de", "ar","nl"]
    if language == "en":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        model = BertModel.from_pretrained('bert-base-uncased', config=config)
    elif language == "de":
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
        config = AutoConfig.from_pretrained("dbmdz/bert-base-german-uncased", output_hidden_states=True)
        model = AutoModel.from_pretrained("dbmdz/bert-base-german-uncased", config=config)
    elif language == "ar":
        tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        config = AutoConfig.from_pretrained("asafaya/bert-base-arabic", output_hidden_states=True)
        model = AutoModel.from_pretrained("asafaya/bert-base-arabic", config=config)
    elif language == "nl":
        tokenizer = BertTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased")
        config = BertConfig.from_pretrained("wietsedv/bert-base-dutch-cased", output_hidden_states=True)
        model = BertModel.from_pretrained("wietsedv/bert-base-dutch-cased", config=config)

    return tokenizer, model


def write_word(token, embds_src, embds_tgt, label, writetime, file_path):
    write_mode = "w" if writetime == 1 else "a"
    with open(file_path+token+".txt", write_mode, encoding='utf-8') as f:
        embds_src = embds_src[np.newaxis,:]
        embds_tgt = embds_tgt[np.newaxis,:]
        f.writelines([label, "\t"])
        np.savetxt(f,embds_src,fmt='%.10f',newline="\t")
        np.savetxt(f,embds_tgt,fmt='%.10f',newline="\n")


def get_sentence(tokens):
    sentence = []
    length = len(tokens)
    i=0
    while(i < length):
        index = [i]
        index = find_pound_key(tokens, i, index)
        i = index[-1]+1
        word = [tokens[j].strip("##") for j in index]
        word = "".join(word)
        sentence.append([word, index])
    return sentence


def find_pound_key(tokens, i, index):
    if tokens[i] == "[SEP]":
        return index
    if "##" not in tokens[i+1]:
        return index

    if "##" in tokens[i+1]:
        index.append(i+1)
        return find_pound_key(tokens, i+1, index)


def get_align_dict(line):
    align = {}
    discard = []
    discard_tgt = []
    seen_tgt = []

    for couple in line:
        couple = couple.split('-')
        c_0 = int(couple[0])
        c_1 = int(couple[1])
        if c_1 in seen_tgt:
            discard_tgt.append(c_1)
        if c_0 not in align:
            align[c_0] = c_1
            seen_tgt.append(c_1)
        else:
            discard.append(c_0)

    for src, tgt in align.items():
        if tgt in discard_tgt:
            discard.append(src)
    discard = set(discard)

    for dis in discard:
        align.pop(dis)

    return align


def checknum(string):
    pattern = re.compile('[0-9]+')
    match = pattern.findall(string)
    if match:
        return True
    else:
        return False


def checkspecial(string):
    test_str = re.search(r"\W",string)
    if test_str==None:
        return False
    else:
        return True


def if_not_in_dict(i, align_dict):
    return True if i not in align_dict else False


def if_special_number(token):
    return True if (checkspecial(token) and len(token)>1) or (checknum(token) and len(token)>1) else False


def if_max(token, token_list, max_num):
    if token in token_list:
        if token_list[token] >= max_num:
            return True
    return False


def if_OOV(token, tokenizer, oov_ind):
    w_id = tokenizer.convert_tokens_to_ids(token)
    if w_id == oov_ind:
        return True
    return False


def if_not_meet_requirement(token, token_list, tokenizer, max_num, oov_ind):
    if if_special_number(token):
        return True
    if if_max(token, token_list, max_num):
        return True
    if if_OOV(token, tokenizer, oov_ind):
        return True
    return False


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--tgt", type=str, required=True)
    parser.add_argument("--open_src_file", type=str, required=True)
    parser.add_argument("--open_tgt_file", type=str, required=True)
    parser.add_argument("--open_align_file", type=str, required=True)
    parser.add_argument("--write_vectors_path", type=str, required=True)

    ## Other parameters
    parser.add_argument("--max_seq_length", default=150, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for predictions.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--max_num_word", type = int, default = 10000)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {} ".format(device, n_gpu))

    tokenizer_src, model_src = get_tokenizer_model(args.src)
    model_src.to(device)

    tokenizer_tgt, model_tgt = get_tokenizer_model(args.tgt)
    model_tgt.to(device)

    oov_ind_src = tokenizer_src.convert_tokens_to_ids('[UNK]')
    oov_ind_tgt = tokenizer_tgt.convert_tokens_to_ids('[UNK]')

    token_list = {}

    # Split large file and process them one-by-one

    logger.info("reading examples from file "+args.open_src_file)   
    examples_src = read_examples(args.open_src_file)
    examples_tgt = read_examples(args.open_tgt_file)

    logger.info("finishing reading.")
    features_src = convert_examples_to_features(
        examples=examples_src, seq_length=args.max_seq_length, tokenizer=tokenizer_src)
    features_tgt = convert_examples_to_features(
        examples=examples_tgt, seq_length=args.max_seq_length, tokenizer=tokenizer_tgt)

    unique_id_to_feature = {}
    for feature in features_src:
        unique_id_to_feature[feature.unique_id] = feature

    all_input_ids_src = torch.tensor([f.input_ids for f in features_src], dtype=torch.long)
    all_input_mask_src = torch.tensor([f.input_mask for f in features_src], dtype=torch.long)
    all_example_index_src = torch.arange(all_input_ids_src.size(0), dtype=torch.long)

    all_input_ids_tgt = torch.tensor([f.input_ids for f in features_tgt], dtype=torch.long)
    all_input_mask_tgt = torch.tensor([f.input_mask for f in features_tgt], dtype=torch.long)
    all_example_index_tgt = torch.arange(all_input_ids_tgt.size(0), dtype=torch.long)

    eval_data_src = TensorDataset(all_input_ids_src, all_input_mask_src, all_example_index_src)
    eval_data_tgt = TensorDataset(all_input_ids_tgt, all_input_mask_tgt, all_example_index_tgt)

    eval_sampler_src = SequentialSampler(eval_data_src)
    eval_sampler_tgt = SequentialSampler(eval_data_tgt)

    eval_dataloader_src = DataLoader(eval_data_src, sampler=eval_sampler_src, batch_size=args.batch_size)
    eval_dataloader_tgt = DataLoader(eval_data_tgt, sampler=eval_sampler_tgt, batch_size=args.batch_size)

    f_align = open(args.open_align_file)

    model_src.eval()
    model_tgt.eval()
    with torch.no_grad():
        for (input_ids_src, input_mask_src, example_indices_src),(input_ids_tgt, input_mask_tgt, example_indices_tgt) in zip(tqdm(eval_dataloader_src), eval_dataloader_tgt):
                input_ids_src = input_ids_src.to(device)
                input_mask_src = input_mask_src.to(device)
                input_ids_tgt = input_ids_tgt.to(device)
                input_mask_tgt = input_mask_tgt.to(device)

                all_encoder_layers_src = model_src(input_ids_src, token_type_ids=None, attention_mask=input_mask_src)[2]
                all_encoder_layers_src = torch.stack(all_encoder_layers_src, dim=0).permute(1,2,0,3)

                all_encoder_layers_tgt = model_tgt(input_ids_tgt, token_type_ids=None, attention_mask=input_mask_tgt)[2]
                all_encoder_layers_tgt = torch.stack(all_encoder_layers_tgt, dim=0).permute(1,2,0,3)

                for b, example_index in enumerate(example_indices_src):

                    line = f_align.readline().split()

                    feature_src = features_src[example_index.item()]
                    sentence_src = get_sentence(feature_src.tokens)
                    feature_tgt = features_tgt[example_index.item()]
                    sentence_tgt = get_sentence(feature_tgt.tokens)
                    length_tgt = len(sentence_tgt)

                    align_dict = get_align_dict(line)

                    for (i, token_src) in enumerate(sentence_src):
                        token_src, index_src = token_src[0], token_src[1]

                        if if_not_meet_requirement(token_src, token_list, tokenizer_src, args.max_num_word, oov_ind_src):
                            continue

                        if if_not_in_dict(i-1, align_dict):
                            token_tgt = "NONE"
                            values_tgt = torch.zeros(1, dtype=float).cpu().numpy()
                        else:
                            pos = align_dict[i-1]+1
                            if pos > length_tgt-1:
                                continue
                            token_tgt = sentence_tgt[pos]
                            token_tgt, index_tgt = token_tgt[0], token_tgt[1]

                            if if_OOV(token_tgt, tokenizer_tgt, oov_ind_tgt) or if_special_number(token_tgt):
                                continue

                            values_tgt = torch.sum(all_encoder_layers_tgt[b,index_tgt,-4:,:], dim=1).sum(dim=0).detach().cpu().numpy()


                        values_src = torch.sum(all_encoder_layers_src[b,index_src,-4:,:], dim=1).sum(dim=0).detach().cpu().numpy()

                        if token_src not in token_list:
                            write_word(token_src, values_src, values_tgt, token_tgt, 1, args.write_vectors_path)
                            token_list[token_src] = 1
                        else:
                            if token_list[token_src] < args.max_num_word:
                                write_word(token_src, values_src, values_tgt, token_tgt, -1, args.write_vectors_path)
                                token_list[token_src] += 1


        f_align.close()
    
if __name__ == "__main__":
    main()
