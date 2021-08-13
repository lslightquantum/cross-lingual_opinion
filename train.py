import argparse
import torch
from utils.train_utils.opinion_expression_utils import train_opinion_expression
from utils.train_utils.srl4orl_utils import train_srl4orl

def main():
    argparser = argparse.ArgumentParser(description='parameters for opinion expression model', 
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('--model_type', type=str, required=True)
    argparser.add_argument('--bert_type', default='bert-base-uncased', type=str)
    argparser.add_argument('--batch_size', default=32, type=int)
    argparser.add_argument('--maxlen', default=80, type=int)
    argparser.add_argument('--rnn_layers', default=3, type=int)
    argparser.add_argument('--rnn_hidden_dim', default=600, type=int)
    argparser.add_argument('--dropout_prob', default=0.5, type=float)
    argparser.add_argument('--device', default='cpu', type=str)
    argparser.add_argument('--lr_base', default=5e-5, type=float)
    argparser.add_argument('--lr_bert', default=3e-5, type=float)
    argparser.add_argument('--grad_clip', default=None, type=float)
    argparser.add_argument('--init_model', default=None, type=str)
    argparser.add_argument('--init_bert_model', default=None, type=str)
    argparser.add_argument('--n_epochs', default=25, type=int)
    argparser.add_argument('--save_best_model', default=False, type=bool)
    argparser.add_argument('--save_path', default=None, type=str)
    argparser.add_argument('--loss_weights', default='fixed_ratio', type=str)
    argparser.add_argument('--loss_weights_ratio', default=6, type=int)
    argparser.add_argument('--train_data', nargs='+')
    argparser.add_argument('--valid_data', nargs='+')
    argparser.add_argument('--srl_train_data', type=str)
    argparser.add_argument('--srl_valid_data', type=str)
    argparser.add_argument('--orl_train_data', type=str)
    argparser.add_argument('--orl_valid_data', type=str)

    config = argparser.parse_args()
    if config.device == 'gpu':
        config.device = 'cuda'
    config.device = torch.device(config.device)

    if config.model_type == 'opinion_expression':
        train_opinion_expression(config)
    elif config.model_type == 'srl4orl':
        train_srl4orl(config)

if __name__ == '__main__':
    main()