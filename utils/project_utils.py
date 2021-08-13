import torch
import itertools

def get_alignment(src, tgt, tokenizer, model, device):
    # pre-processing
    # sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    sent_src, sent_tgt = src, tgt
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    ids_src, ids_tgt = ids_src.to(device), ids_tgt.to(device)
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
        align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )

    return align_words

def _get_aligned_tags(alignment, src_tags, tgt_len, default_tag='O'):
    align_words = alignment
    tgt_tags = [default_tag] * tgt_len
    for i, j in sorted(align_words):
        if tgt_tags[j] == default_tag:
            tgt_tags[j] = src_tags[i]
    return tgt_tags

def get_aligned_tags(src, tgt, src_tags, tokenizer, model, device):
    alignment = get_alignment(src, tgt, tokenizer, model, device)
    tgt_len = len(tgt)
    tgt_tags = _get_aligned_tags(alignment, src_tags, tgt_len, default_tag='O')

    return tgt_tags