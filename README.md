# Cross-lingual Opinion Mining

## train models
### opinion expression labeling model
```
python train.py --model_type opinion_expression --train_data <path to training data> --valid_data <path to validation data>
```
### opinion role labeling model
```
python train.py --model_type srl4orl --srl_train_data <path to srl training data> --srl_valid_data <path to srl validation data> --orl_train_data <path to orl training data> --orl_valid_data <path to orl validation data>
```
### more (optional) options
#### set device, set pre-trained BERT model, and save model with best validation score
```
--device <"cpu" or "gpu"/"cuda"> --bert_type <name or path to bert model> --save_best_model True --save_path <output model path>
```
#### hyperparameters
```
--batch_size 32 --maxlen 80 --rnn_layers 3 --rnn_hidden_dim 600 --dropout_prob 0.5 --lr_base 5e-5 --lr_bert 3e-5 --grad_clip 1 --n_epochs 25 --loss_weights fixed_ratio --loss_weights_ratio 6
```

## translate annotated data and project annotation
### data translation
Translate input sentences with google translation api and tokenize the results with spacy.
```
python translate.py --input_file <source file path> --output_file <output file path> --source_langauge <language of source sentences> --target_language <target language> --google_credential_json <path to google api credential json file>
```
### annotation projection
Project annotations to translated data with [awesome-align](https://github.com/neulab/awesome-align). By default, the model uses "bert-base-multilingual-cased". You can also download fine-tuned models from [awesome-align](https://github.com/neulab/awesome-align).
```
python project.py --input_file <source file path> --output_file <output file path> --bert_model <name or path to bert model>
```
