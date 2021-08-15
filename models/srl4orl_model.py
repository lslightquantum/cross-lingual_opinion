import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from models.CRF_module import CRF

class SRL4ORL(nn.Module):
    def __init__(self, bert_type, srl_tagset_size, orl_tagset_size, rnn_layers, hidden_dim, init_model=None):
        super(SRL4ORL, self).__init__()
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_dim
        self.srl_tagset_size = srl_tagset_size
        self.orl_tagset_size = orl_tagset_size

        if not init_model:
            self.bert = BertModel.from_pretrained(bert_type)
        else:
            self.bert = BertModel(BertConfig.from_json_file(bert_type))
        self.token_embedding_dim = self.bert.config.hidden_size
        self.ctx_embedding_dim = self.token_embedding_dim
        self.p_embedding = nn.Embedding(2, self.ctx_embedding_dim)
        self.rnn = nn.LSTM(self.token_embedding_dim+self.ctx_embedding_dim, hidden_dim // 2, batch_first=True, 
                           num_layers=self.rnn_layers, bidirectional=True, dropout=0.0)
        self.srl_hidden2tag = nn.Linear(hidden_dim, self.srl_tagset_size)
        self.orl_hidden2tag = nn.Linear(hidden_dim, self.orl_tagset_size)
        self.srl_crf = CRF(num_tags=self.srl_tagset_size, batch_first=True)
        self.orl_crf = CRF(num_tags=self.orl_tagset_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        if init_model:
            self.load_state_dict(torch.load(init_model))
    
    def _get_token_embedds(self, nn_input, mask, word_head):
        token_embedds = self.bert(nn_input, mask).last_hidden_state
        reduced_token_embedds = [token_embedds[i][word_head[i][:len(token_embedds[i])]>0, :] for i in range(len(token_embedds))]
        max_len = max([item.shape[0] for item in reduced_token_embedds])
        reduced_mask = torch.tensor([[1]*len(item) + 
                                     [0]*(max_len - len(item)) 
                                     for item in reduced_token_embedds]).to(nn_input.device)
        reduced_token_embedds = torch.stack([torch.cat((item, 
                                                        torch.zeros(max_len - item.shape[0], 
                                                                    item.shape[1]).to(nn_input.device)), 
                                                       dim=0) for item in reduced_token_embedds])

        return reduced_token_embedds, reduced_mask.bool()

    def _build_rnn_input(self, task, token_embedds, mask, p):
        p = p[:, :token_embedds.shape[1]].long()
        ctx_embedds = self.p_embedding(p)
        rnn_in = torch.cat([token_embedds, ctx_embedds], dim=-1)

        return rnn_in

    def _run_rnn(self, nn_input, mask):
        seq_lens = mask.sum(dim=-1)
        packed_input = nn.utils.rnn.pack_padded_sequence(nn_input, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_rnn_out, _ = self.rnn(packed_input)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rnn_out, batch_first=True)
        rnn_out = self.dropout(rnn_out)

        return rnn_out

    def get_emissions(self, task, nn_input, mask, word_head, p):
        token_embedds, reduced_mask = self._get_token_embedds(nn_input, mask, word_head)
        rnn_in = self._build_rnn_input(task, token_embedds, reduced_mask, p)
        rnn_out = self._run_rnn(rnn_in, reduced_mask)
        if task == 'srl':
            emissions = self.srl_hidden2tag(rnn_out)
        elif task =='orl':
            emissions = self.orl_hidden2tag(rnn_out)
        else:
            print('task should be either "srl" or "orl"')
        return emissions, reduced_mask

    def decode(self, task, emissions, mask):
        if task == 'srl':
            tag_seq = self.srl_crf.decode(emissions, mask)
        elif task == 'orl':
            tag_seq = self.orl_crf.decode(emissions, mask)
        else:
            print('task should be either "srl" or "orl"')
        return tag_seq
        
    def neg_log_likelihood(self, task, emissions, tags, mask, reduction="mean"):
        if task == 'srl':
            loss = -self.srl_crf(emissions, tags, mask, reduction="mean")
        elif task == 'orl':
            loss = -self.orl_crf(emissions, tags, mask, reduction="mean")
        else:
            print('task should be either "srl" or "orl"')
        return loss
