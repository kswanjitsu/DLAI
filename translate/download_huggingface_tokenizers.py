from tokenizers import BertWordPieceTokenizer
import urllib
from transformers import AutoTokenizer
import os

def download_vocab_files_for_tokenizer(tokenizer, model_type, output_path):
    vocab_files_map = tokenizer.pretrained_vocab_files_map
    vocab_files = {}
    for resource in vocab_files_map.keys():
        download_location = vocab_files_map[resource][model_type]
        f_path = os.path.join(output_path, os.path.basename(download_location))
        urllib.request.urlretrieve(download_location, f_path)
        vocab_files[resource] = f_path
    return vocab_files

model_type = 'bert-base-uncased'
output_path = './my_local_vocab_files/'
tokenizer = AutoTokenizer.from_pretrained(model_type)
vocab_files = download_vocab_files_for_tokenizer(tokenizer, model_type, output_path)
fast_tokenizer = BertWordPieceTokenizer(vocab_files.get('vocab_file'), vocab_files.get('merges_file'))