def get_token_label_from_spans(char_idx, spans):
    label = 'O'
    for span in spans:
        start = span['start']
        end = span['end']
        if start <= char_idx <= end:
            head = 'I'
            if char_idx == start:
                head = 'B'
            label = head + '-' + span['label']
            break

    return label


def convert_spacy_jsonl_to_bio(input_path, output_path, spacy_path):
    import jsonlines
    import spacy
    from tqdm import tqdm
    print('loading spacy model...')
    nlp = spacy.load(spacy_path)
    print('writing data...')
    with jsonlines.open(input_path) as reader:
        with open(output_path, 'w') as writer:
            for obj in tqdm(reader):
                spans = obj['spans'] if 'spans' in obj else []
                doc = nlp(obj['text'])
                for t in doc:
                    token = t.text.replace(' ', '')
                    if len(token) > 0:
                        label = get_token_label_from_spans(t.idx, spans) if not t.is_punct else 'O'
                        writer.write(token + ' ' + label + '\n')
                writer.write('\n')


def split_all_bio_data_into_three_sets(train=0.65, test=0.25, dev=0.1, data_path='all.txt'):
    import random
    from tqdm import tqdm

    def write_sentence(writer, sent):
        for l in sent:
            writer.write(l)
        writer.write('\n')

    sents = []
    sent = []
    r = open(data_path, 'r')
    for l in tqdm(r):
        if l == '\n' and len(sent) > 0:
            sents.append(sent.copy())
            sent = []
        else:
            sent.append(l)
    r.close()

    random.shuffle(sents)

    w1 = open('train.txt', 'w')
    w2 = open('test.txt', 'w')
    w3 = open('dev.txt', 'w')
    train_threshold = int(train * len(sents))
    test_threshold = train_threshold + int(test * len(sents))
    count = 0
    for s in tqdm(sents):
        if count <= train_threshold:
            write_sentence(w1, s)
        elif count <= test_threshold:
            write_sentence(w2, s)
        else:
            write_sentence(w3, s)
        count += 1
    w1.close()
    w2.close()
    w3.close()


def extract_labels(data_path='all.txt', save_label_path='labels.txt'):
    r = open(data_path, 'r')
    labels = {}
    for l in r:
        l = l.replace('\n', '')
        if len(l) > 0:
            label = l.split(' ')[1]
            if label in labels:
                labels[label] = labels[label] + 1
            else:
                labels[label] = 1
    r.close()
    w = open(save_label_path, 'w')
    for lbl in labels.keys():
        w.write(lbl + '\n')
    w.close()
    return labels


# map the original NER labels as follows:
# PERSON -> PER
# PRODUCT -> MISC
# GPE -> LOC
# EVENT -> MISC
def map_original_labels_to_conventional(data_path='all.txt'):
    lines = []
    r = open(data_path, 'r')
    for l in r:
        l = l.replace('PERSON', 'PER').replace('PRODUCT', 'MISC')\
            .replace('GPE', 'LOC').replace('EVENT', 'MISC')
        lines.append(l)
    r.close()
    w = open(data_path, 'w')
    for l in lines:
        w.write(l)
    w.close()
