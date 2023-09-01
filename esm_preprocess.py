import torch


def preprocess_esm():

    esm_model, alphabet = torch.hub.load("facebookresearch/esm", "esm_if1_gvp4_t16_142M_UR50")
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3", "K A <mask> I S Q"),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    print(batch_lens)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    # Look at the unsupervised self-attention map contact predictions
    import matplotlib.pyplot as plt
    for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
        plt.matshow(attention_contacts[: tokens_len, : tokens_len])
        plt.title(seq)
        plt.show()


if __name__ == '__main__':
    preprocess_esm()
