import os
import torch
import nltk

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.build_dictionary( [
            os.path.join(path, 'train.txt'),
            os.path.join(path, 'valid.txt'),
            os.path.join(path, 'test.txt') ])
        
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def build_dictionary(self, paths, max_size=16000):
        allWordDist = nltk.FreqDist()
        
        for path in paths:
            assert os.path.exists(path)

            with open(path, 'r', encoding="utf8") as f:
                words = nltk.tokenize.word_tokenize(f.read())
                wordDist = nltk.FreqDist([word.lower() for word in words])
                del words
                allWordDist.update(wordDist)
                del wordDist
                
        for word, _ in allWordDist.most_common(max_size):
            self.dictionary.add_word(word)
        
        self.dictionary.add_word('<unk>')
        self.dictionary.add_word('<eos>')
        print('Vocabulary Size:', len(self.dictionary))
        
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = []
            for i, line in enumerate(f):
                words = nltk.word_tokenize(line) + ['<eos>']
                for word in words:
                    word = word.lower()
                    if word not in self.dictionary.word2idx:
                        word = '<unk>'
                    ids.append(self.dictionary.word2idx[word])

        return torch.LongTensor(ids)
