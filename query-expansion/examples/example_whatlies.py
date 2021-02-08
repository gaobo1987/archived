import spacy
from whatlies import Embedding
import matplotlib.pylab as plt


def lookup(word, vocab) -> list:
    _id = vocab.strings[str(word)]
    _vec = vocab.vectors[_id]
    return _vec


def compose_tokens(words, vocab):
    tokens = {}
    for w in words:
        vec = lookup(w, vocab)
        tokens[w] = Embedding(w, vec)
    return tokens

en = {}
zh = spacy.load('models/wiki_zh_align')
s = '你好,这是一个中文句子。'
doc = zh(s)
for t in doc:
    print(t.text, t.pos_, t.lemma_, t.has_vector)


# words_en = ['dockor', 'nurse', 'king', 'queen', 'beautiful', 'brave', 'man', 'woman', 'tiger', 'rabbit']
# words_zh = ['医生', '护士', '国王', '皇后', '美丽', '勇敢', '男人', '女人', '老虎', '兔子']
words_en = ['king', 'queen', 'man', 'woman']
words_zh = ['国王', '皇后', '男人', '女人']


tokens = compose_tokens(words_en, en.vocab)
tokens_zh = compose_tokens(words_zh, zh.vocab)
tokens.update(tokens_zh)

x_axis = tokens['男人']
y_axis = tokens['女人']
for name, t in tokens.items():
    t.plot(x_axis=x_axis, y_axis=y_axis, color='red').plot(x_axis=x_axis, y_axis=y_axis, kind='text')
