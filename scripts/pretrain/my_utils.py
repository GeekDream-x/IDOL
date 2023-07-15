import os
import sys
import json
import torch
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as pltx
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, RobertaTokenizerFast, AlbertTokenizerFast, DebertaV2Tokenizer
from transformers.models.bert.modeling_bert import BertForLogicPreTraining
from transformers.models.roberta.modeling_roberta import RobertaForLogicMaskedLM
from transformers.models.albert.modeling_albert import AlbertForLogicPreTraining
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2ForLogicMaskedLM

def is_existed_rec(name,obj):
    if isinstance(obj,list):
        for s in obj:
            assert(os.path.exists(s)),f"{name}:{s} does not exist"
            fsize = os.path.getsize(s)/float(1024*1024)
            print(s,round(fsize,2))
    elif isinstance(obj,str):
        assert(os.path.exists(obj)),f"{name}:{obj} does not exist"
        fsize = os.path.getsize(obj)/float(1024*1024)
        print(obj,round(fsize,2))
    elif isinstance(obj,dict):
        for k,v in obj.items():
            is_existed_rec(name+'-'+k,v)
    else:
        print(f"Unknown object:{name}:{obj}")
        return

def read_data_json(filename: str):
    with open(filename,'r') as f:
        data_json = json.load(f)
    #check all files exist
    for k,v in data_json.items():
        is_existed_rec(k,v)
    return data_json

MODEL_CLASS_MAPPING = {
            "bert": BertForLogicPreTraining,
            "roberta": RobertaForLogicMaskedLM,
            "albert": AlbertForLogicPreTraining,
            "deberta": DebertaV2ForLogicMaskedLM,
        }

class LogicTextDataset(Dataset):
    def __init__(self, files_dir, max_len, tokenizer):
        
        self.max_len = max_len
        self.samples = []
        for file_path in glob(f"{files_dir}/*.jsonl"):
            with open(file_path, 'r') as f:
                self.samples.extend(list(f.readlines()))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = json.loads(self.samples[idx])
        text, logic_polarity_label = sample['text'], sample['logic_polarity_label']

        encode_dict= self.tokenizer(text, padding='max_length', max_length=self.max_len, truncation=True)

        if isinstance(self.tokenizer, RobertaTokenizerFast):
            input_ids, attention_mask = encode_dict['input_ids'], encode_dict['attention_mask']
        else:
            input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']

        logic_polarity_label = [-100] + logic_polarity_label
        
        while len(logic_polarity_label) < len(input_ids):
            logic_polarity_label.append(-100)
        while len(logic_polarity_label) > len(input_ids):
            logic_polarity_label.pop()
        

        if isinstance(self.tokenizer, RobertaTokenizerFast):
            input_ids, attention_mask, logic_polarity_label = torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(logic_polarity_label)
            return_dict =  dict({
                "input_ids":input_ids,
                "attention_mask":attention_mask,
                "logic_polarity_labels":logic_polarity_label
            })
        else:
            input_ids, token_type_ids, attention_mask, logic_polarity_label = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(logic_polarity_label)
            return_dict =  dict({
                "input_ids":input_ids,
                "token_type_ids":token_type_ids,
                "attention_mask":attention_mask,
                "logic_polarity_labels":logic_polarity_label
            })

        return return_dict

