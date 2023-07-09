import torch
from transformers import LlamaTokenizer
from model.modeling_llama import LlamaForCausalLM
from transformers.modeling_utils import PreTrainedModel
import json
from tqdm import tqdm
import traceback

def encode_captions(src_path, dst_path, model_name):
    llama_tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
    llama_model = LlamaForCausalLM.from_pretrained(model_name)
    id2emb = dict()

    with open(src_path, 'r') as json_file:
        data = json.load(json_file)

    for key in tqdm(data.keys(), desc='Encoding captions'):
        text = data[key]
        tokens = llama_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        tokens = tokens.input_ids
        emb = llama_model.model.embed_tokens(tokens)
        id2emb[key] = emb

    torch.save(id2emb, dst_path + '/id2emb.pt')

if __name__ == '__main__':

    # src_path = '../caption-pdbs/abstract.json'
    # dst_path = '../caption-pdbs'

    # try:
    #     a = []
    #     b = a[1]
    # except Exception as e:
    #     traceback.print_exc()

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('test preprocess on device:', device)

        model_name = 'lmsys/vicuna-13b-v1.3'

        print('start to test model:', model_name)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
        llama_model = LlamaForCausalLM.from_pretrained(model_name)
        print('Successfully loaded llm model:', model_name)

        abstract_path = './../caption-pdbs/abstract.json'
        with open(abstract_path, 'r') as json_file:
            data = json.load(json_file)

        dict_to_save = dict()
        for entry in tqdm(data, desc="Encoding captions: "):
            id = entry['pdb_id']
            text = entry['caption']

            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False, max_length=512, padding='max_length')
            attn_mask = tokens.attention_mask.to(device)
            tokens = tokens.input_ids.to(device)
            emb = llama_model.model.embed_tokens(tokens)
            dict_to_save[id] = emb

        torch.save(dict_to_save, './../caption-pdbs/id2emb.pt')
        print('Successfully saved id2emb.pt')

    except Exception as e:
        print('something wrong:')
        traceback.print_exc()


