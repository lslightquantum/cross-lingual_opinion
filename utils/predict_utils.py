import os, json
from tqdm.auto import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizerFast
from models.opinion_expression_model import  OpinionDetector
from models.srl4orl_model import SRL4ORL

def read_sentences(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    sentences = [line.strip() for line in lines]
    
    return sentences

def tokenize_sentences(sentences, tokenizer):
    tokenized_sentences = ['\t'.join(tokenizer(sentence).tokens()[1:-1]).replace('\t##', '').split('\t') for sentence in sentences]
    return tokenized_sentences

def encode_sentence(sentence, tokenizer):
    tokens = [tokenizer.cls_token]
    word_head_ids = []
    for word in sentence:
        word_head_ids.append(len(tokens))
        tokens += tokenizer.tokenize(word)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids.append(tokenizer.sep_token_id)
    word_head = [0]*len(input_ids)
    for j in range(len(word_head_ids)):
        word_head[word_head_ids[j]] = 1

    return input_ids, word_head

def get_inference_dataloader(sentences, tokenizer, batch_size=32):

    def collate_fn(batch):
        batch_input_ids = torch.stack([item[0] for item in batch], dim=0)
        batch_mask = torch.stack([item[1] for item in batch], dim=0)
        batch_word_head = torch.stack([item[2] for item in batch], dim=0)
        label_maxlen = batch_word_head.sum(dim=-1).max()
        return batch_input_ids, batch_mask, batch_word_head

    X = []
    attention_mask = []
    word_head = []
    for sentence in sentences:
        tokenized_sentence, sentence_word_head = encode_sentence(sentence, tokenizer) 
        X.append(torch.tensor(tokenized_sentence))
        attention_mask.append(torch.ones(len(tokenized_sentence)))
        word_head.append(torch.tensor(sentence_word_head))
    X = pad_sequence(X, padding_value=tokenizer.pad_token_id, batch_first=True)
    attention_mask = pad_sequence(attention_mask, padding_value=0.0, batch_first=True)
    word_head = pad_sequence(word_head, padding_value=0, batch_first=True)
    dataset = TensorDataset(X, attention_mask, word_head)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)

    return dataloader

def load_model(task, model_path, device):
    model_path = model_path
    with open(os.path.join(model_path, 'config.json')) as f:
        model_config = json.load(f)
    if task == 'opinion':
        with open(os.path.join(model_path, 'idx2tag.json')) as f:
            idx2tag = json.load(f)
        idx2tag = {int(k): v for k, v in idx2tag.items()}
        tagset_size = model_config['tagset_size']
    elif task == 'srl4orl':
        with open(os.path.join(model_path, 'srl_idx2tag.json')) as f:
            srl_idx2tag = json.load(f)
        srl_idx2tag = {int(k): v for k, v in srl_idx2tag.items()}
        srl_tagset_size = model_config['srl_tagset_size']
        with open(os.path.join(model_path, 'orl_idx2tag.json')) as f:
            orl_idx2tag = json.load(f)
        orl_idx2tag = {int(k): v for k, v in orl_idx2tag.items()}
        orl_tagset_size = model_config['orl_tagset_size']
        idx2tag = {'srl': srl_idx2tag, 'orl': orl_idx2tag}
    bert_type = os.path.join(model_path, 'bert_config.json')
    rnn_layers = model_config['rnn_layers']
    rnn_hidden_dim = model_config['hidden_dim']
    init_model = os.path.join(model_path, 'state_dict.pt')
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(model_path, "tokenizer"), return_tensors="pt")
    if task == 'opinion':
        model = OpinionDetector(bert_type, tagset_size, rnn_layers, rnn_hidden_dim, init_model=init_model).to(device)
    elif task == 'srl4orl':
        model = SRL4ORL(bert_type, srl_tagset_size, orl_tagset_size, rnn_layers, rnn_hidden_dim, init_model=init_model).to(device)

    return {'model': model, 'tokenizer': tokenizer, 'idx2tag': idx2tag}

def get_tags_list(outputs, batch_mask):
    batch_tags = outputs.argmax(dim=-1)

    tags_list = []
    for tags, mask in zip(batch_tags, batch_mask):
        tags_list.append(tags[:mask.sum()].tolist())

    return tags_list

def get_spans(name, tags, idx2tag, scheme='bmoes'):
    if scheme == 'bmoes':
        return _get_spans_bmoes(name, tags, idx2tag)
    elif scheme == 'bio':
        return _get_spans_bio(name, tags, idx2tag)

def _get_spans_bmoes(name, tags, idx2tag):
    spans = []
    begin_id = -1
    end_id = -1
    for i in range(len(tags)):
        tag = idx2tag.get(int(tags[i]), 'None')
        if begin_id >= 0 and not tag.startswith('M-' + name) and not tag.startswith('E-' + name):
            begin_id = -1
        if tag.startswith('S-' + name):
            spans.append([i, i+1])
            begin_id = -1
        elif tag.startswith('B-' + name):
            begin_id = i
        elif tag.startswith('E-' + name) and begin_id >= 0:
            end_id = i + 1
            spans.append([begin_id, end_id])

    return spans

def _get_spans_bio(name, tags, idx2tag):
    spans = []
    begin_id = -1
    end_id = -1
    for i in range(len(tags)):
        tag = idx2tag.get(int(tags[i]), 'None')

        if (tag.startswith('B-') or 
            tag.startswith('O') or
            (tag.startswith('I-') and not tag.startswith('I-' + name))) and begin_id >= 0:

            end_id = i
            spans.append([begin_id, end_id])
            begin_id = -1

        if tag.startswith('B-' + name):
            begin_id = i

    return spans

def opinion_inference_batch(model, batch, idx2tag, device):
    model.eval()
    batch_spans = []
    with torch.no_grad():
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        word_head = batch[2].to(device)

        batch_outputs, batch_mask = model(input_ids, attention_mask, word_head)
        batch_outputs = get_tags_list(batch_outputs, batch_mask)

    for tags in batch_outputs:
        neg_spans = get_spans("DSE-neg", tags, idx2tag, scheme="bio")
        pos_spans = get_spans("DSE-pos", tags, idx2tag, scheme="bio")
        batch_spans.append([neg_spans, pos_spans])

    return batch_spans

def orl_inference_batch(model, batch, idx2tag, device):
    model.eval()
    batch_spans = []
    with torch.no_grad():
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        word_head = batch[2].to(device)
        p = batch[3].to(device)

        emissions, batch_mask = model.get_emissions('orl', input_ids, attention_mask, word_head, p)
        batch_outputs = model.decode('orl', emissions, batch_mask)

    for tags in batch_outputs:
        hdr_spans = get_spans("h", tags, idx2tag, scheme="bio")
        tgt_spans = get_spans("t", tags, idx2tag, scheme="bio")
        batch_spans.append([hdr_spans, tgt_spans])

    return batch_spans

def make_srl4orl_batch(original_batch, batch_ds_spans, srl4orl_tokenizer):
    input_ids = []
    attention_mask = []
    word_head = []
    p = []
    num_dss = []

    for i in range(len(batch_ds_spans)):
        ds_spans = batch_ds_spans[i]
        num_dup = len(ds_spans)
        num_dss.append(num_dup)
        input_ids += [original_batch[0][i]] * num_dup
        attention_mask += [original_batch[1][i]] * num_dup
        word_head += [original_batch[2][i]] * num_dup
        num_words = original_batch[1][i].sum().long()
        for ds_span in ds_spans:
            p_temp = torch.zeros((num_words)).long()
            p_temp[ds_span[0]:ds_span[1]] = 1
            p.append(p_temp)

    if len(input_ids) > 0:
        input_ids = pad_sequence(input_ids, padding_value=srl4orl_tokenizer.pad_token_id, batch_first=True)
        attention_mask = pad_sequence(attention_mask, padding_value=-1, batch_first=True)
        word_head = pad_sequence(word_head, padding_value=0, batch_first=True)
        p = pad_sequence(p, padding_value=0, batch_first=True).long()
        data = (input_ids, attention_mask, word_head, p)
    else:
        data = None

    return {'data': data, 'num_dss': num_dss}

def inference(opinion_model, srl4orl_model, dataloader, device, tqdm_leave=False):
    opinion_model['model'].eval()
    srl4orl_model['model'].eval()
    opinion_spans = []
    opinion_role_spans = []
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=tqdm_leave):
            batch_spans = opinion_inference_batch(opinion_model['model'], batch, opinion_model['idx2tag'], device)
            opinion_spans += batch_spans
            batch_ds_spans = [[span for polar_spans in spans for span in polar_spans] for spans in batch_spans]
            batch = make_srl4orl_batch(batch, batch_ds_spans, srl4orl_model['tokenizer'])
            if batch['data']:
                batch_spans = orl_inference_batch(srl4orl_model['model'], batch['data'], srl4orl_model['idx2tag']['orl'], device)
            else:
                batch_spans = []
            grouped_batch_spans = []
            sid = 0
            for num_dss in batch['num_dss']:
                grouped_batch_spans.append(batch_spans[sid:sid + num_dss])
                sid = sid + num_dss
            opinion_role_spans += grouped_batch_spans

    return opinion_spans, opinion_role_spans

def format_output(sentences, tokenized_sentences, opinion_spans, opinion_role_spans):
    formatted_output = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        tokenized_sentence = tokenized_sentences[i]
        frames = []
        frame_id = 0
        for j in range(2):
            polarity = ['negative', 'positive'][j]
            ds_spans = opinion_spans[i][j]
            for span in ds_spans:
                expression = {'indices': list(range(span[0], span[1])), 'tokens': tokenized_sentence[span[0]:span[1]]}
                frame = {'polarity': polarity, 'expression': expression, 'holders': [], 'targets': []}
                hdr_spans = opinion_role_spans[i][frame_id][0]
                tgt_spans = opinion_role_spans[i][frame_id][1]
                frame_id += 1
                for span in hdr_spans:
                    frame['holders'].append({'indices': list(range(span[0], span[1])), 'tokens': tokenized_sentence[span[0]:span[1]]})
                for span in tgt_spans:
                    frame['targets'].append({'indices': list(range(span[0], span[1])), 'tokens': tokenized_sentence[span[0]:span[1]]})
                frames.append(frame)
        formatted_output.append({'opinion_frames': frames, 'sentence': sentence, 'tokenized': tokenized_sentence})
    
    return formatted_output