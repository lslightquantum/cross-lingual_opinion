from tqdm.auto import tqdm

def read_txt_data(data_path):
    X = []
    y = []

    with open(data_path) as f:
        lines = f.readlines()

    tokens = []
    tags = []
    for line in lines:
        line = line.strip()
        if not line:
            X.append(tokens)
            y.append(tags)   
            tokens = []
            tags = []
        else:
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)

    return X, y

def translate_text(source, target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    import six
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, source_language=source, target_language=target)

    source_text = result["input"]
    translated_text = result["translatedText"]

    return source_text, translated_text

def translate_all(source, target, sentences):
    lines = []
    for sentence in tqdm(sentences):
        s_source, s_target = translate_text(source, target, sentence)
        lines.append([s_source, s_target])
    return lines