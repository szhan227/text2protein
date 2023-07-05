import torch
from transformers import LlamaTokenizer
from model.modeling_llama import LlamaForCausalLM
import json

def encode_captions(src_path, dst_path, model_name):
    llama_tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
    llama_model = LlamaForCausalLM.from_pretrained(model_name)
    id2emb = dict()

    with open(src_path, 'r') as json_file:
        data = json.load(json_file)

    for key in data.keys():
        text = data[key]
        tokens = llama_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        tokens = tokens.input_ids

        emb = llama_model.model.embed_tokens(tokens)
        id2emb[key] = emb

    torch.save(id2emb, dst_path + '/id2emb.pt')

if __name__ == '__main__':

    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-13b-v1.3', use_fast=False)

    text = 'Fibroblast growth factors (FGFs) are key regulators of cell proliferation, tumor-induced angiogenesis, and migration. FGFs are essential for early embryonic development, organ formation, and angiogenesis. FGF1 also plays an important role in inflammation, wound healing, and restenosis. The biological effects of FGF1 are mediated through the activation of the four transmembrane phosphotyrosine kinase fibroblast growth factor receptors in the presence of heparin sulfate proteoglycans and, therefore, require the release of the protein into the extracellular space. FGF1 is exported through a non-classical release pathway involving the formation of a specific multiprotein complex. The protein constituents of this complex include FGF1, S100A13, and the p40 form of synaptotagmin 1 (Syt1). Because FGF1 plays an important role in tumor formation, it is clear that preventing the formation of the multiprotein complex would be an effective strategy to inhibit a wide range of cancers. To understand the molecular events in the FGF1 release pathway, we studied the FGF1-S100A13 tetrameric and FGF1-S100A13-C2A hexameric complex structures, which are both complexes possibly formed during the non-classical pathway of FGF1 release.'
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    tokens = tokens.input_ids
    print('tokens shape:', tokens.shape)

    llama_model = LlamaForCausalLM.from_pretrained('lmsys/vicuna-13b-v1.3')
    emb = llama_model.model.embed_tokens(tokens)
    print('emb:', emb)
    print('emb shape:', emb.shape)


