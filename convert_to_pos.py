# -*- coding: utf-8 -*-
"""
INSTRUCTIONS

python convert_to_pos.py --source=<author>

For example:

    python convert_to_pos.py --source=collins

will convert documents within ./data/collins into respective POS translated
documents

Output:
./data/<auth>_pos with pos text files
"""

import os, re
from nltk import tag
from nltk import tokenize

def convert_punctuation(pos):
    if pos == '.':
        return "PERIOD"
    if pos == ',':
        return 'COMMA'
    if pos == ':':
        return 'COLON'
    if pos == ';':
        return 'SEMICOLON'
    if pos == '?':
        return 'QUES'
    if pos == '!':
        return "EXCLA"
    if pos == "''" or pos == '``' or pos == '"':
        return 'QUOTE'
    if pos == '-NONE-':
        return ''
    return pos


def get_paths(directory_name):
    return ['./data/{0}/{1}'.format(directory_name, i)
            for i in os.listdir('./data/{0}/'.format(directory_name))
            if i != '.DS_Store']


def convert_to_pos(input_directory_name):
    output_directory = './data/{0}_pos'.format(input_directory_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    input_paths = get_paths(input_directory_name)
    for path in input_paths:
        print 'converted {0}'.format(path)
        f = open(path)
        text = unicode(f.read(), 'utf-8').encode('ascii', 'ignore')
        text_pos = ''
        for sentence in tokenize.sent_tokenize(text):
            tokenized_sentence = tokenize.word_tokenize(sentence)
            pos_sentence = tag.pos_tag(tokenized_sentence)
            text_pos += ' ' + ' '.join([convert_punctuation(w[1])
                                        for w in pos_sentence])
        f_pos = open('./data/{0}_pos/{1}'.format(input_directory_name,
                                                 path.split('/')[-1]), 'w')
        f_pos.write(text_pos)
        f_pos.flush()
        f_pos.close()

