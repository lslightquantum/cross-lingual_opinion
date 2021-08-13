import torch
import transformers
import json, argparse
from tqdm.auto import tqdm
from utils.project_utils import get_aligned_tags

def main():
    argparser = argparse.ArgumentParser(description='parameters for opinion expression model', 
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('--input_file', type=str, required=True)
    argparser.add_argument('--output_file', type=str, required=True)
    argparser.add_argument('--bert_model', type=str, default='bert-base-multilingual-cased')

    config = argparser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config.input_file) as f:
        lines = json.load(f)

    model = transformers.BertModel.from_pretrained(config.bert_model).to(device)
    tokenizer = transformers.BertTokenizer.from_pretrained(config.bert_model)

    for line in tqdm(lines):
        src = line['tokens_source']
        tgt = line['tokens_target']
        src_tags = line['tags_source']
        tgt_tags = get_aligned_tags(src, tgt, src_tags, tokenizer, model, device)
        line['tags_target'] = tgt_tags

    with open(config.output_file, 'w') as f:
        json.dump(lines, f)

if __name__ == '__main__':
    main()