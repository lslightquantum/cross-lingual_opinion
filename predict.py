import argparse
import torch
from utils.predict_utils import *

def main():
    argparser = argparse.ArgumentParser(description='parameters for opinion expression model', 
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('--input_file', type=str, required=True)
    argparser.add_argument('--output_file', type=str, required=True)
    argparser.add_argument('--opinion_expression_model', type=str, required=True)
    argparser.add_argument('--srl4orl_model', type=str, required=True)

    config = argparser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opinion_model = load_model('opinion', config.opinion_expression_model, device)
    srl4orl_model = load_model('srl4orl', config.srl4orl_model, device)

    sentences = read_sentences(config.input_file)
    tokenized_sentences = tokenize_sentences(sentences, opinion_model['tokenizer'])
    dataloader = get_inference_dataloader(tokenized_sentences, opinion_model['tokenizer'], batch_size=32)
    opinion_spans, opinion_role_spans = inference(opinion_model, srl4orl_model, dataloader, device, tqdm_leave=False)
    formatted_output = format_output(sentences, tokenized_sentences, opinion_spans, opinion_role_spans)

    with open(config.output_file, 'w') as f:
        json.dump(formatted_output, f)
    

if __name__ == '__main__':
    main()