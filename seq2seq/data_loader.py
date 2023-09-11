from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
import torch.utils.data
from transformations import tensors_from_pair

def str2bool(v):
    return v.lower() in ("yes", "true")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 1
EOS_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<pad>", SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub("[.!?]", '', s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1, lang2, auto_encoder=False, reverse=False):
    print("Reading lines...")

    lines = open('./data/%s-%s.txt' % ('eng', 'fra'), encoding='utf-8'). \
        read().strip().split('\n')

    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    if auto_encoder:
        pairs = [[pair[0], pair[0]] for pair in pairs]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p, max_input_length):
    return len(p[0].split(' ')) < max_input_length and \
           len(p[1].split(' ')) < max_input_length and \
           p[1].startswith(eng_prefixes)

def filter_pairs(pairs, max_input_length):
    pairs = [pair for pair in pairs if filter_pair(pair, max_input_length)]
    return pairs

def prepare_data(lang1, lang2, max_input_length, auto_encoder=False, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, auto_encoder, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs, max_input_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

class Dataset():
    def __init__(self, phase, num_embeddings=None, max_input_length=None, transform=None, auto_encoder=False):
        if auto_encoder:
            lang_in = 'eng'
            lang_out = 'eng'
        else:
            lang_in = 'eng'
            lang_out = 'fra'

        input_lang, output_lang, pairs = prepare_data(lang_in, lang_out, max_input_length, auto_encoder=auto_encoder, reverse=True)
        print(random.choice(pairs))

        random.shuffle(pairs)

        if phase == 'train':
            selected_pairs = pairs[0:int(0.8 * len(pairs))]
        else:
            selected_pairs = pairs[int(0.8 * len(pairs)):]

        selected_pairs_tensors = [tensors_from_pair(selected_pairs[i], input_lang, output_lang, max_input_length)
                     for i in range(len(selected_pairs))]

        self.transform = transform
        self.num_embeddings = num_embeddings
        self.max_input_length = max_input_length
        self.data = selected_pairs_tensors
        self.input_lang = input_lang
        self.output_lang = output_lang

    def langs(self):
        return self.input_lang, self.output_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        sample = {'sentence': pair}

        if self.transform:
            sample = self.transform(sample)

        return sample
