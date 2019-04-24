import pycantonese as pc
import os
import re

# Get Dictionary from hkcancor

corpus = pc.hkcancor()
freq = corpus.word_frequency()

def save(file_path, init_words_path, tagged_words):

    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(init_words_path, 'r', encoding='utf-8') as t:
        lines = t.readlines()
        with open(file_path, 'w', encoding='utf-8') as f:
            for word in corpus.tagged_words():
                word_freq = freq[word[0]] if word[0] in freq else None
                word_tag = word[1].lower()
                word_tag_matched = bool(re.match('^[a-z]+$', word_tag))
                word_line = word[0]
                if word_freq is not None:
                    word_line = word_line + ' ' + str(word_freq)
                if word_tag_matched is True:
                    word_line = word_line + ' ' + str(word_tag)
                f.write(word_line + '\n')

            for line in lines:
                f.write(line)

save('./data/dict.txt', './data/init_dict.txt', corpus.tagged_words())

# Include the Stop Word with dictionary in pycantonese

'''
with open('StopWord.txt', 'w', encoding='utf-8') as StopWord:
    stop_words = pc.stop_words()
    for word in stop_words:
        StopWord.write(word + '\n')
'''