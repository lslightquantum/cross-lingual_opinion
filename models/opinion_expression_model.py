import torch
from torch import nn
from transformers import BertModel, BertConfig

class OpinionDetector(nn.Module):
    def __init__(self, bert_type, tagset_size, rnn_layers, hidden_dim, init_model=None):
        super(OpinionDetector, self).__init__()
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        
        if not init_model:
            self.bert = BertModel.from_pretrained(bert_type)
        else:
            self.bert = BertModel(BertConfig.from_pretrained(bert_type))
        self.embedding_dim = self.bert.config.hidden_size
        self.rnn = nn.GRU(self.embedding_dim, hidden_dim // 2, batch_first=True, 
                           num_layers=self.rnn_layers, bidirectional=True, dropout=0.0)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.dropout = nn.Dropout(p=0.5)
        if init_model:
            self.load_state_dict(torch.load(init_model))

    def forward(self, nn_input, mask, word_head):
        bert_feats = self.bert(nn_input, mask).last_hidden_state
        bert_feats = self.dropout(bert_feats)
        seq_lens = mask.sum(dim=-1)
        bert_feats = nn.utils.rnn.pack_padded_sequence(bert_feats, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(bert_feats)
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        reduced_rnn_out = [rnn_out[i][word_head[i][:len(rnn_out[i])]>0, :] for i in range(len(rnn_out))]
        max_len = max([item.shape[0] for item in reduced_rnn_out])
        model_device = next(self.parameters()).device
        reduced_mask = torch.tensor([[1]*len(item) + 
                                     [0]*(max_len - len(item)) 
                                     for item in reduced_rnn_out]).to(model_device)
        reduced_rnn_out = torch.stack([torch.cat((item, 
                                                   torch.zeros(max_len - item.shape[0], 
                                                               item.shape[1]).to(model_device)), 
                                                  dim=0) for item in reduced_rnn_out])
        logits = self.hidden2tag(reduced_rnn_out)

        return logits, reduced_mask.byte()