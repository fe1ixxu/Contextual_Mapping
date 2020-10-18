import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, required=True)
parser.add_argument("--tgt", type=str, required=True)
parser.add_argument("--src_file", type=str, required=True)
parser.add_argument("--tgt_file", type=str, required=True)
parser.add_argument("--write_src_file", type=str, required=True)
parser.add_argument("--write_tgt_file", type=str, required=True)

args = parser.parse_args()

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
        sentence.append(word)
    return sentence

def find_pound_key(tokens, i, index):
    if i == len(tokens)-1:
        return index
    if "##" not in tokens[i+1]:
        return index
    if "##" in tokens[i+1]:
        index.append(i+1)
        return find_pound_key(tokens, i+1, index)

def get_tokenizer(language):
    assert language in ["en", "de", "fi", "ar", "it", "el", "nl", "es", "pl", "ro","sv", "pt", "bg"]
    if language == "en":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif language == "de":
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
    elif language == "fi":
        tokenizer = BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-uncased-v1")
    elif language == "ar":
        tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
    elif language == "it":
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
    elif language == "el":
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
    elif language == "nl":
        tokenizer = BertTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased")
    elif language == "es":
        tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    elif language == "pl":
        tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
    elif language == "ro":
        tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1", do_lower_case=True)
    elif language == "sv":
        tokenizer = AutoTokenizer.from_pretrained("af-ai-center/bert-base-swedish-uncased")
    elif language == "pt":
        tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased", do_lower_case=True)
    elif language == "bg":
        tokenizer = AutoTokenizer.from_pretrained("anon-submission-mk/bert-base-macedonian-bulgarian-cased", do_lower_case=True)

    return tokenizer

tokenizer_src = get_tokenizer(args.src)
tokenizer_tgt = get_tokenizer(args.tgt)

print("start tokenizing src..... at", args.write_src_file)
num = 1000000
i=0
with open(args.src_file) as f: #"./en.txt"
    with open(args.write_src_file, "w") as f2:   #"./en_token.txt"
        line = f.readline()
        while(line):
            tokenized_text = tokenizer_src.tokenize(line)
            tokenized_text = get_sentence(tokenized_text)
            f2.writelines([" ".join(tokenized_text) ,'\n'])
            line = f.readline()
            i += 1
            if i == num:
                break
print("finished...")

print("start tokenizing tgt..... at", args.write_tgt_file)

i=0         
with open(args.tgt_file) as f:
    with open(args.write_tgt_file, "w") as f2:
        line = f.readline()
        while(line):
            tokenized_text = tokenizer_tgt.tokenize(line)
            tokenized_text = get_sentence(tokenized_text)
            f2.writelines([" ".join(tokenized_text) ,'\n'])
            line = f.readline()
            i += 1
            if i == num:
                break
print("finished...")