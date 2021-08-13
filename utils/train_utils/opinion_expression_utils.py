import numpy as np
from numpy import random
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from collections import OrderedDict, Counter
from tqdm.auto import tqdm
from pathlib import Path
import os, json
from os import path
from models.opinion_expression_model import  OpinionDetector

def train_opinion_expression(config):
    device = config.device
    train_files = config.train_data
    valid_files = config.valid_data
    train_data = load_data(train_files[0])
    valid_data = load_data(valid_files[0])
    for train_file, valid_file in zip(train_files[1:], valid_files[1:]):
        extra_train_data = load_data(train_file)
        extra_valid_data = load_data(valid_file)
        for k, v in extra_train_data.items():
            train_data[k] += v
        for k, v in extra_valid_data.items():
            valid_data[k] += v

    vocab = build_vocab(train_files[0])
    word2idx, tag2idx, idx2word, idx2tag = build_idx_map(vocab)

    tagset_size = len(tag2idx)
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_type, return_tensors="pt")

    train_dataloader = get_dataloader(train_data['sentences'], train_data['labels'], 
                                      tokenizer, tag2idx, config.batch_size)
    eval_dataloader = get_dataloader(valid_data['sentences'], valid_data['labels'], 
                                     tokenizer, tag2idx, config.batch_size)

    if config.loss_weights == None:
        criterion = nn.CrossEntropyLoss(ignore_index = -1)
    else:
        if config.loss_weights == 'fixed_ratio':
            tag_weights = {tag: (config.loss_weights_ratio if tag != 'O' else 1) for tag in idx2tag.values()}
            weight = torch.tensor([tag_weights[tag] if tag in tag_weights else 0 for tag in tag2idx.keys()], dtype=torch.float)
            weight = (weight/weight.sum()).to(device)
        elif config.loss_weights == 'inverse_frequency':
            weight = get_inverse_frequency_weights(train_dataloader).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index = -1)

    model = OpinionDetector(config.bert_type, tagset_size, config.rnn_layers, 
                            config.rnn_hidden_dim, init_model=config.init_model).to(device)
    if config.init_bert_model:
        model.bert.load_state_dict(torch.load(config.init_bert_model))

    params = list(map(lambda x: x[1],list(filter(lambda kv: 'bert' in kv[0], model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: 'bert' not in kv[0], model.named_parameters()))))

    optimizer = AdamW([{'params': base_params}, 
                       {'params': params, 'lr': config.lr_bert}], lr=config.lr_base)

    summary = train_and_evaluate(model, criterion, optimizer, train_dataloader, eval_dataloader, 
                                 tokenizer, config.n_epochs, idx2tag, device, 
                                 grad_clip=config.grad_clip, schedule_lr=True, mixed_precision=(device.type=='cuda'),
                                 save_best_model=config.save_best_model, save_path=config.save_path)

def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path) as file:
        words = []
        tags = []
        for line in file:
            line = line.strip().split('\t')
            if not len(line) == 2 and words:
                sentences.append(words)
                labels.append(tags)
                words = []
                tags = []
            elif len(line) == 2:
                words.append(line[0])
                tags.append(line[1])
    return {'sentences': sentences, 'labels': labels}

def build_vocab(file_path):
    words = set()
    tags = set()
    with open(file_path) as file:
        for line in file:
            line = line.strip().split('\t')
            if len(line) == 2:
                words.add(line[0])
                tags.add(line[1])
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

def get_dataloader(sentences, tags, tokenizer, tag2idx, batch_size=32, maxlen=80):

    def collate_fn(batch):
        batch_input_ids = torch.stack([item[0] for item in batch], dim=0)
        batch_labels = torch.stack([item[1] for item in batch], dim=0)
        batch_mask = torch.stack([item[2] for item in batch], dim=0)
        batch_word_head = torch.stack([item[3] for item in batch], dim=0)
        label_maxlen = batch_word_head.sum(dim=-1).max()
        batch_labels = batch_labels[:, :label_maxlen]
        return batch_input_ids, batch_labels, batch_mask, batch_word_head

    X = []
    y = []
    attention_mask = []
    word_head = []
    for sentence, text_labels in zip(sentences, tags):
        tokenized_sentence, sentence_word_head = encode_sentence(sentence, tokenizer)
        if len(tokenized_sentence) <= maxlen:
            labels = [tag2idx[tag] for tag in text_labels]
            X.append(torch.tensor(tokenized_sentence))
            y.append(torch.tensor(labels))
            attention_mask.append(torch.ones(len(tokenized_sentence)))
            word_head.append(torch.tensor(sentence_word_head))
    X = pad_sequence(X, padding_value=tokenizer.pad_token_id, batch_first=True)
    y = pad_sequence(y, padding_value=-1, batch_first=True)
    attention_mask = pad_sequence(attention_mask, padding_value=0.0, batch_first=True)
    word_head = pad_sequence(word_head, padding_value=0, batch_first=True)
    dataset = TensorDataset(X, y, attention_mask, word_head)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

    return dataloader

def train_epoch(model, criterion, optimizer, data_loader, idx2tag, device, grad_clip=None, schedule_lr=False, scheduler=None, mixed_precision=False, scaler=None):
    model.train()
    losses = []
    label_types = set([item[item.index('-')+1:] for item in idx2tag.values() if item != 'O' ])
    label_types.add('micro')
    f1_metrics = {mode: {label_type: np.zeros((3)) for label_type in label_types} for mode in ['bin', 'prop']}

    tqdm_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=False, disable=False)
    for i, data in tqdm_iterator:
        input_ids = data[0].to(device)
        attention_mask = data[2].to(device)
        labels = data[1].to(device)
        word_head = data[3].to(device)

        if mixed_precision:
            with torch.cuda.amp.autocast():
                outputs, batch_mask = model(input_ids, attention_mask, word_head)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                outputs = get_tags_list(outputs, batch_mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if schedule_lr:
                scheduler.step()
        else:
            outputs, batch_mask = model(input_ids, attention_mask, word_head)
            if device.type == 'cpu':
                outputs = outputs.contiguous()
                labels = labels.contiguous()
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            outputs = get_tags_list(outputs, batch_mask)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if schedule_lr:
                scheduler.step()

        losses.append(loss.item())
        labels = labels.data.cpu().numpy().flatten()
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

def evaluate_epoch(model, criterion, data_loader, idx2tag, device):
    model.eval()
    losses = []
    label_types = set([item[item.index('-')+1:] for item in idx2tag.values() if item != 'O' ])
    label_types.add('micro')
    f1_metrics = {mode: {label_type: np.zeros((3)) for label_type in label_types} for mode in ['bin', 'prop']}

    tqdm_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=False, disable=False)
    with torch.no_grad():
        for i, data in tqdm_iterator:
            input_ids = data[0].to(device)
            attention_mask = data[2].to(device)
            labels = data[1].to(device)
            word_head = data[3].to(device)

            outputs, batch_mask = model(input_ids, attention_mask, word_head)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            outputs = get_tags_list(outputs, batch_mask)
            losses.append(loss.item())
            labels = labels.data.cpu().numpy().flatten()
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

def train_and_evaluate(model, criterion, optimizer,
                       train_dataloader, eval_dataloader, 
                       tokenizer, n_epochs,
                       idx2tag, device, grad_clip=None, 
                       schedule_lr=False, mixed_precision=False,
                       save_best_model=False, save_path=None):
    summary = {'train': [], 'eval': []}

    train_datalen = len(train_dataloader)

    scheduler = None
    if schedule_lr:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                                    num_training_steps=train_datalen*n_epochs)
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
                     'tagset_size': model.tagset_size,
                     'rnn_layers': model.rnn_layers,
                     'hidden_dim': model.hidden_dim}
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(model_cfg, f)
        with open(os.path.join(save_path, 'idx2tag.json'), 'w') as f:
            json.dump(idx2tag, f)

        
    for i in range(1, n_epochs + 1):
        print('-'*150)
        print('epoch ', i, ':')
        
        train_result = train_epoch(model, criterion, optimizer, train_dataloader, idx2tag, device, 
                                   grad_clip, schedule_lr, scheduler, mixed_precision, scaler)
        
        train_loss = train_result['loss']
        train_f1_bin = train_result['f1s']['bin']['micro']
        train_f1_prop = train_result['f1s']['prop']['micro']
        summary['train'].append(train_result)
        print('TRAIN || loss: {:.4f}; f1 bin: {:.4f}; f1 prop: {:.4f}'.format(train_loss, train_f1_bin, train_f1_prop))

        eval_result = evaluate_epoch(model, criterion, eval_dataloader, idx2tag, device)
        eval_loss = eval_result['loss']
        eval_f1_bin = eval_result['f1s']['bin']['micro']
        eval_f1_prop = eval_result['f1s']['prop']['micro']
        summary['eval'].append(eval_result)
        print('VALID || loss: {:.4f}; f1 bin: {:.4f}; f1 prop: {:.4f}'.format(eval_loss, eval_f1_bin, eval_f1_prop))

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

def pad_outputs(outputs):
    max_len = max([len(seq) for seq in outputs])
    rv = [seq + [-1]*(max_len - len(seq)) for seq in outputs]
    return rv

def accuracy(labels, outputs):
    mask = (labels >= 0)

    return np.sum(labels[mask] == outputs[mask]), np.sum(mask)

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

def get_tags_list(outputs, batch_mask):
    batch_tags = outputs.argmax(dim=-1)

    tags_list = []
    for tags, mask in zip(batch_tags, batch_mask):
        tags_list.append(tags[:mask.sum()].tolist())

    return tags_list