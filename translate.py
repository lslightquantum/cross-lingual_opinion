from nltk.tokenize.moses import MosesDetokenizer
import re, argparse, json, os
from utils.translate_utils import read_txt_data, translate_all
import nltk, spacy

def main():
    argparser = argparse.ArgumentParser(description='parameters for opinion expression model', 
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('--input_file', type=str, required=True)
    argparser.add_argument('--output_file', type=str, required=True)
    argparser.add_argument('--source_language', default=None, type=str)
    argparser.add_argument('--target_language', type=str, required=True)
    argparser.add_argument('--google_credential_json', default='', type=str)

    config = argparser.parse_args()

    ### download nltk package
    nltk.download('perluniprops')

    ### load spacy package
    if config.target_language == 'zh-cn':
        spacy_package = "zh_core_web_sm"
    elif config.target_language == 'en':
        spacy_package = "en_core_web_sm"
    else:
        spacy_package = config.target_language + "_core_news_sm"

    try:
        nlp = spacy.load(spacy_package)
    except:
        print("spaCy: target language is not downloaded or not supported.")
        exit()

    ### set google appication credentials environment variable if provided
    if config.google_credential_json:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.google_credential_json

    ### load data
    X, y = read_txt_data(config.input_file)
    
    ### detokenize sentences
    detokenizer = MosesDetokenizer()
    tokensub = {"`": "'", "``": '"',
                "''": '"'}
    sentences = []
    for tokens, tags in zip(X, y):
        tokens = [tokensub[token] if token in tokensub else token for token in tokens]
        sentence = detokenizer.detokenize(tokens, return_str=True)
        sentence = sentence.replace('( ', '(').replace('[ ', '[').replace('{ ', '{ ').replace(" n't", "n't").replace(', "', '," ').replace('. "', '."')
        sentences.append(sentence)

    ### translate
    translated_sentences = translate_all(config.source_language, config.target_language, sentences)

    ### format output
    formatted_data = []
    for i in range(len(X)):
        sentence_source = translated_sentences[i][0]
        sentence_target = translated_sentences[i][1]
        sentence_target = re.sub(r'\（[A-z0-9-.;&# ]*\）', '', sentence_target)
        tokens_source = X[i]
        tags_source = y[i]

        formatted_data.append({'sentence_source': sentence_source,
                               'sentence_target': sentence_target,
                               'tokens_source': tokens_source,
                               'tags_source': tags_source})

    ### save data before tokenizing in case tokenization fails
    with open(config.output_file, 'w') as f:
        json.dump(formatted_data, f)

    ### tokenize translated sentences
    nlp = spacy.load(spacy_package)
    for item in formatted_data:
        sentence_target = item['sentence_target']
        doc = nlp(sentence_target)
        tokens_target = [token.text for token in doc]
        item['tokens_target'] = tokens_target

    ### save data
    with open(config.output_file, 'w') as f:
        json.dump(formatted_data, f)

if __name__ == '__main__':
    main()