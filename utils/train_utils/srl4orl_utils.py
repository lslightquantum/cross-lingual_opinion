import numpy as np
from numpy import random
from tqdm.auto import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from collections import OrderedDict, Counter
from pathlib import Path
import os, json
from torch.optim import AdamW
from models.srl4orl_model import SRL4ORL

def train_srl4orl(config):
    device = config.device
    srl_train = load_data(config.srl_train_data)
    orl_train = load_data(config.orl_train_data)
    orl_valid = load_data(config.orl_valid_data)

    srl_vocab = build_vocab(config.srl_train_data)
    orl_vocab = build_vocab(config.orl_train_data)
    _, srl_tag2idx, _, srl_idx2tag = build_idx_map(srl_vocab)
    _, orl_tag2idx, _, orl_idx2tag = build_idx_map(orl_vocab)

    srl_tagset_size = len(srl_tag2idx)
    orl_tagset_size = len(orl_tag2idx)

    tokenizer = BertTokenizerFast.from_pretrained(config.bert_type, return_tensors="pt")
    # srl_train['sentences'], srl_train['labels'] = srl_train['sentences'][:1000], srl_train['labels'][:1000]
    srl_train_dataloader = get_dataloader(srl_train['sentences'], srl_train['labels'], tokenizer, srl_tag2idx, config.batch_size, task='srl')
    orl_train_dataloader = get_dataloader(orl_train['sentences'], orl_train['labels'], tokenizer, orl_tag2idx, config.batch_size, task='orl')
    orl_valid_dataloader = get_dataloader(orl_valid['sentences'], orl_valid['labels'], tokenizer, orl_tag2idx, config.batch_size, task='orl')

    model = SRL4ORL(config.bert_type, srl_tagset_size, orl_tagset_size, 
                    config.rnn_layers, config.rnn_hidden_dim, init_model=config.init_model).to(device)
    params = list(map(lambda x: x[1],list(filter(lambda kv: 'bert' in kv[0], model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: 'bert' not in kv[0], model.named_parameters()))))
    optimizer = AdamW([{'params': base_params}, 
                    {'params': params, 'lr': config.lr_bert}], lr=config.lr_base)
    
    result = train_and_evaluate(model, optimizer, srl_train_dataloader, 
                                orl_train_dataloader, orl_valid_dataloader, 
                                tokenizer, config.n_epochs,
                                srl_idx2tag, orl_idx2tag, device, 
                                grad_clip=config.grad_clip, 
                                schedule_lr=True, mixed_precision=(device.type=='cuda'),
                                save_best_model=config.save_best_model, save_path=config.save_path)


def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path) as file:
        words = []
        tags = []
        for line in file:
            line = line.strip().split('\t')
            if len(line) == 1 and words:
                sentences.append(words)
                labels.append(tags)
                words = []
                tags = []
            else:
                words.append(line[0])
                tags.append(line[1])
    return {'sentences': sentences, 'labels': labels}

def build_vocab(file_path):
    words = set()
    tags = set()
    with open(file_path) as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line) > 1:
                words.add(line[0])
                tags.add(line[1])
    # tags.add('PAD_TAG')
    return {'words': words, 'tags': tags}

def build_idx_map(vocab):
    word2idx = {}
    for i, word in enumerate(vocab['words']):
        word2idx[word] = i
    idx2word = {idx: word for word, idx in word2idx.items()}

    tag2idx = {}
    for i, tag in enumerate(vocab['tags']):
        tag2idx[tag] = i
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return word2idx, tag2idx, idx2word, idx2tag

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

def get_dataloader(sentences, tags, tokenizer, tag2idx, batch_size=32, maxlen=80, task='srl', window_size=1):

    def collate_fn(batch):
        batch_input_ids = torch.stack([item[0] for item in batch], dim=0)
        batch_labels = torch.stack([item[1] for item in batch], dim=0)
        batch_p = torch.stack([item[2] for item in batch], dim=0)
        batch_m = torch.stack([item[3] for item in batch], dim=0)
        batch_mask = torch.stack([item[4] for item in batch], dim=0)
        batch_word_head = torch.stack([item[5] for item in batch], dim=0)
        label_maxlen = batch_word_head.sum(dim=-1).max()
        batch_labels = batch_labels[:, :label_maxlen]
        return batch_input_ids, batch_labels, batch_p, batch_m, batch_mask, batch_word_head

    if task == 'srl':
        p_tag = 'V'
    elif task == 'orl':
        p_tag = 'ds'

    X = []
    y = []
    p = []
    m = []
    attention_mask = []
    word_head = []
    for sentence, text_labels in tqdm(zip(sentences, tags), leave=False, total=len(sentences)):
        tokenized_sentence, sentence_word_head = encode_sentence(sentence, tokenizer)
        if len(tokenized_sentence) <= maxlen:
            labels = [tag2idx[tag] if not tag.endswith(p_tag) else tag2idx['O'] for tag in text_labels]
            p_temp = [idx for idx in range(len(text_labels)) if text_labels[idx].endswith(p_tag)]
            if p_temp:
                p_start, p_end = p_temp[0], p_temp[-1] + 1
                ctx_start, ctx_end = max(0, p_temp[0] - window_size), min(len(text_labels), p_temp[-1] + 1 + window_size)
                p_mark = [1 if idx >= p_start and idx < p_end else 0 for idx in range(len(text_labels))]
                region_mark = [1 if idx >= ctx_start and idx < ctx_end else 0 for idx in range(len(text_labels))]
            else:
                p_mark = [0] * len(text_labels)
                region_mark = [0] * len(text_labels)
            X.append(torch.tensor(tokenized_sentence))
            y.append(torch.tensor(labels))
            p.append(torch.tensor(p_mark))
            m.append(torch.tensor(region_mark))
            attention_mask.append(torch.ones(len(tokenized_sentence)))
            word_head.append(torch.tensor(sentence_word_head))
    X = pad_sequence(X, padding_value=tokenizer.pad_token_id, batch_first=True)
    y = pad_sequence(y, padding_value=-1, batch_first=True)
    p = pad_sequence(p, padding_value=0, batch_first=True).bool()
    m = pad_sequence(m, padding_value=0, batch_first=True).bool()
    attention_mask = pad_sequence(attention_mask, padding_value=0.0, batch_first=True)
    word_head = pad_sequence(word_head, padding_value=0, batch_first=True)
    dataset = TensorDataset(X, y, p, m, attention_mask, word_head)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

    return dataloader

def train_epoch(model, optimizer, task, data_loader, idx2tag, device, grad_clip=None, schedule_lr=False, scheduler=None, mixed_precision=False, scaler=None):
    model.train()
    losses = []
    label_types = set([item[item.index('-')+1:] for item in idx2tag.values() if item != 'O' ])
    label_types.add('micro')
    if task == 'srl':
        label_types.remove('V')
    elif task == 'orl':
        label_types.remove('ds')
    f1_metrics = {mode: {label_type: np.zeros((3)) for label_type in label_types} for mode in ['bin', 'prop']}


    tqdm_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=False, disable=False)
    for i, data in tqdm_iterator:
        batch_input_ids = data[0].to(device)
        batch_labels = data[1].to(device)
        batch_p = data[2].to(device)
        batch_m = data[3].to(device)
        batch_mask = data[4].to(device)
        batch_word_head = data[5].to(device)

        if mixed_precision:
            with torch.cuda.amp.autocast():
                emissions, batch_mask = model.get_emissions(task, batch_input_ids, batch_mask, batch_word_head, batch_p, batch_m)
                loss = model.neg_log_likelihood(task, emissions, batch_labels, batch_mask, reduction="mean")
                outputs = model.decode(task, emissions, batch_mask)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if schedule_lr:
                scheduler.step()
        else:
            emissions, batch_mask = model.get_emissions(task, batch_input_ids, batch_mask, batch_word_head, batch_p, batch_m)
            loss = model.neg_log_likelihood(task, emissions, batch_labels, batch_mask, reduction="mean")
            outputs = model.decode(task, emissions, batch_mask)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if schedule_lr:
                scheduler.step()

        losses.append(loss.item())
        labels = batch_labels.data.cpu().numpy().flatten()
        outputs = np.array(pad_outputs(outputs)).flatten()

        for label_type in label_types:
            if label_type != 'micro':
                for mode in ['bin', 'prop']:
                    batch_f1_metrics = get_batch_f1_metrics(label_type, outputs, labels, idx2tag, span_scheme="bio", f1_mode=mode)
                    f1_metrics[mode][label_type] += batch_f1_metrics
                    f1_metrics[mode]['micro'] += batch_f1_metrics

        train_loss = float(np.mean(losses))
        train_f1_bin = f1(f1_metrics['bin']['micro'])
        train_f1_prop = f1(f1_metrics['prop']['micro'])
        tqdm_iterator.set_postfix(loss='{:.4f}'.format(train_loss), f1_bin='{:.4f}'.format(train_f1_bin), f1_prop='{:.4f}'.format(train_f1_prop))

    f1s = {mode: {label_type: 0 for label_type in label_types} for mode in ['bin', 'prop']}
    for label_type in label_types:
        for mode in ['bin', 'prop']:
            f1s[mode][label_type] = f1(f1_metrics[mode][label_type])

    return {'loss': train_loss, 'f1s': f1s, 'f1_metrics': f1_metrics}

def train_epoch_both(model, optimizer, srl_dataloader, orl_dataloader, srl_idx2tag, orl_idx2tag, device,
                     grad_clip=None, schedule_lr=False, scheduler=None, mixed_precision=False, scaler=None):
    srl_result = train_epoch(model, optimizer, 'srl', srl_dataloader, srl_idx2tag, device, 
                             grad_clip=grad_clip, schedule_lr=schedule_lr, scheduler=scheduler, 
                             mixed_precision=mixed_precision, scaler=scaler)
    for i in range(2):
        orl_result = train_epoch(model, optimizer, 'orl', orl_dataloader, orl_idx2tag, device,
                                grad_clip=grad_clip, schedule_lr=schedule_lr, scheduler=scheduler, 
                                mixed_precision=mixed_precision, scaler=scaler)

    return {'srl': srl_result, 'orl': orl_result}

def evaluate_epoch(model, task, data_loader, idx2tag, device):
    model.eval()
    losses = []
    label_types = set([item[item.index('-')+1:] for item in idx2tag.values() if item != 'O' ])
    label_types.add('micro')
    if task == 'srl':
        label_types.remove('V')
    elif task == 'orl':
        label_types.remove('ds')
    f1_metrics = {mode: {label_type: np.zeros((3)) for label_type in label_types} for mode in ['bin', 'prop']}

    tqdm_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=False, disable=False)
    for i, data in tqdm_iterator:
        batch_input_ids = data[0].to(device)
        batch_labels = data[1].to(device)
        batch_p = data[2].to(device)
        batch_m = data[3].to(device)
        batch_mask = data[4].to(device)
        batch_word_head = data[5].to(device)

        emissions, batch_mask = model.get_emissions(task, batch_input_ids, batch_mask, batch_word_head, batch_p, batch_m)
        loss = model.neg_log_likelihood(task, emissions, batch_labels, batch_mask, reduction="mean")
        outputs = model.decode(task, emissions, batch_mask)

        losses.append(loss.item())
        labels = batch_labels.data.cpu().numpy().flatten()
        outputs = np.array(pad_outputs(outputs)).flatten()

        for label_type in label_types:
            if label_type != 'micro':
                for mode in ['bin', 'prop']:
                    batch_f1_metrics = get_batch_f1_metrics(label_type, outputs, labels, idx2tag, span_scheme="bio", f1_mode=mode)
                    f1_metrics[mode][label_type] += batch_f1_metrics
                    f1_metrics[mode]['micro'] += batch_f1_metrics

        eval_loss = float(np.mean(losses))
        eval_f1_bin = f1(f1_metrics['bin']['micro'])
        eval_f1_prop = f1(f1_metrics['prop']['micro'])
        tqdm_iterator.set_postfix(loss='{:.4f}'.format(eval_loss), f1_bin='{:.4f}'.format(eval_f1_bin), f1_prop='{:.4f}'.format(eval_f1_prop))

    f1s = {mode: {label_type: 0 for label_type in label_types} for mode in ['bin', 'prop']}
    for label_type in label_types:
        for mode in ['bin', 'prop']:
            f1s[mode][label_type] = f1(f1_metrics[mode][label_type])

    return {'loss': eval_loss, 'f1s': f1s, 'f1_metrics': f1_metrics}

def train_and_evaluate(model, optimizer, srl_train_dataloader, 
                       orl_train_dataloader, orl_valid_dataloader, 
                       tokenizer, n_epochs,
                       srl_idx2tag, orl_idx2tag, device, grad_clip=None, 
                       schedule_lr=False, mixed_precision=False,
                       save_best_model=False, save_path=None):
    summary = {'train': [], 'eval': []}

    scheduler = None
    if schedule_lr:
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
        #                                             num_training_steps=(len(srl_train_dataloader) + len(orl_train_dataloader))*n_epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                                    num_training_steps=(len(orl_train_dataloader))*n_epochs)
    scaler = None
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    best_eval_f1_bin = 0
    best_epoch = 0

    if save_best_model:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(os.path.join(save_path, 'tokenizer'))
        model.bert.config.to_json_file(os.path.join(save_path, 'bert_config.json'))
        model_cfg = {'bert_type': model.bert.name_or_path, 
                     'srl_tagset_size': model.srl_tagset_size,
                     'orl_tagset_size': model.orl_tagset_size,
                     'rnn_layers': model.rnn_layers,
                     'hidden_dim': model.hidden_dim}
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(model_cfg, f)
        with open(os.path.join(save_path, 'srl_idx2tag.json'), 'w') as f:
            json.dump(srl_idx2tag, f)            
        with open(os.path.join(save_path, 'orl_idx2tag.json'), 'w') as f:
            json.dump(orl_idx2tag, f)
        
    for i in range(1, n_epochs + 1):
        print('-'*150)
        print('epoch ', i, ':')
        
        train_result = train_epoch_both(model, optimizer, srl_train_dataloader, orl_train_dataloader, srl_idx2tag, orl_idx2tag, 
                                        device, grad_clip, schedule_lr, scheduler, mixed_precision, scaler)
        
        orl_train_loss = train_result['orl']['loss']
        train_f1_bin_hdr = train_result['orl']['f1s']['bin']['h']
        train_f1_bin_tgt = train_result['orl']['f1s']['bin']['t']
        train_f1_prop_hdr = train_result['orl']['f1s']['prop']['h']
        train_f1_prop_tgt = train_result['orl']['f1s']['prop']['t']
        summary['train'].append(train_result)
        print('TRAIN || loss: {:.4f}; holder f1 bin: {:.4f}; holder f1 prop: {:.4f}; target f1 bin: {:.4f}; target f1 prop: {:.4f}'.format(orl_train_loss, train_f1_bin_hdr, train_f1_prop_hdr, train_f1_bin_tgt, train_f1_prop_tgt))
        
        eval_result = evaluate_epoch(model, 'orl', orl_valid_dataloader, orl_idx2tag, device)
        orl_eval_loss = eval_result['loss']
        eval_f1_bin = eval_result['f1s']['bin']['micro']
        eval_f1_bin_hdr = eval_result['f1s']['bin']['h']
        eval_f1_bin_tgt = eval_result['f1s']['bin']['t']
        eval_f1_prop_hdr = eval_result['f1s']['prop']['h']
        eval_f1_prop_tgt = eval_result['f1s']['prop']['t']
        summary['eval'].append(eval_result)
        print('VALID || loss: {:.4f}; holder f1 bin: {:.4f}; holder f1 prop: {:.4f}; target f1 bin: {:.4f}; target f1 prop: {:.4f}'.format(orl_eval_loss, eval_f1_bin_hdr, eval_f1_prop_hdr, eval_f1_bin_tgt, eval_f1_prop_tgt))

        if eval_f1_bin > best_eval_f1_bin:
            best_epoch = i
            if save_best_model:
                best_eval_f1_bin = eval_f1_bin
                state_dict = OrderedDict()
                for k, v in model.state_dict().items():
                    state_dict[k] = v.cpu()
                
                torch.save(state_dict, os.path.join(save_path, 'state_dict.pt'))
        
    print('-'*150)
    summary['best_epoch'] = best_epoch

    return summary

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


def _overlap(span0, span1):
    if span0[0] < span1[1] and span1[0] < span0[1]:
        return min([span0[1] - span1[0], span1[1] - span0[0]])
    else:
        return -1

def _has_overlap(span0, span_list):
    for span1 in span_list:
        overlap = _overlap(span0, span1)
        if overlap > 0:
            return overlap
    return 0

def _f1(output_spans, label_spans, mode='bin'):
    if mode == 'bin':
        return _f1_bin(output_spans, label_spans)
    elif mode == 'prop':
        return _f1_prop(output_spans, label_spans)
    elif mode == 'exact':
        return _f1_exact(output_spans, label_spans)

def _f1_bin(output_spans, label_spans):
    tp_len, pred_len, true_len = 0, 0, 0
    for span in output_spans:
        overlap = _has_overlap(span, label_spans)
        if overlap > 0:
            tp_len += 1
        pred_len += 1
    for span in label_spans:
        true_len += 1
    
    return np.array([tp_len, pred_len, true_len])

def _f1_prop(output_spans, label_spans):
    tp_len, pred_len, true_len = 0, 0, 0
    for span in output_spans:
        pred_len += 1
    for span in label_spans:
        overlap = _has_overlap(span, output_spans)
        if overlap > 0:
            tp_len += overlap/(span[1] - span[0])
        true_len += 1
    
    return np.array([tp_len, pred_len, true_len])

def _f1_exact(output_spans, label_spans):
    tp_len, pred_len, true_len = 0, 0, 0
    for span in output_spans:
        if span in label_spans:
            tp_len += 1
        pred_len += 1
    for span in label_spans:
        true_len += 1
    
    return np.array([tp_len, pred_len, true_len])

def get_batch_f1_metrics(label_type, outputs, labels, idx2tag, span_scheme, f1_mode):
    batch_f1_metrics = np.zeros((3))

    output_spans = get_spans(label_type, outputs, idx2tag, scheme=span_scheme)
    label_spans = get_spans(label_type, labels, idx2tag, scheme=span_scheme)

    batch_f1_metrics += _f1(output_spans, label_spans, mode=f1_mode)

    return batch_f1_metrics


def f1(f1_metrics):
    tp_len, pred_len, true_len = f1_metrics
    precision = tp_len/pred_len if pred_len != 0 else 0
    recall = tp_len/true_len if true_len !=0 else 0
    f1_score = 2 * precision*recall / (precision + recall) if (precision + recall)!= 0 else 0

    return f1_score

def pad_outputs(outputs):
    max_len = max([len(seq) for seq in outputs])
    rv = [seq + [-1]*(max_len - len(seq)) for seq in outputs]
    return rv

def get_inverse_frequency_weights(dataloader):
    all_tags = []
    for batch in train_dataloader:
        for tags in batch[1].tolist():
            all_tags += tags
    tag_counter = Counter(all_tags)
    del tag_counter[-1]
    for k in tag_counter:
        tag_counter[k] = 1/tag_counter[k]
    total = sum(tag_counter.values())
    for k in tag_counter:
        tag_counter[k] /= total
    weights = []
    for i in range(len(tag_counter)):
        weights.append(tag_counter[i])
    weights = torch.tensor(weights)

    return weights