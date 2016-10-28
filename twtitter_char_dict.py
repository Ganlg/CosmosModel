import pickle

# MEANINGLESS_CHAR = \
#     (
#         '\t', ' ', '!', '#', '$', '%', '&', "'", '(', ')',
#         '*', '+', ',', '-', '.', '/',':', ';', '=', '?',
#         '@', '[', '\\', ']', '^', '_', '`','{', '|', '}',
#         '~','¤', '§', '¨', '°', '´', '¸'
#     )

# generate character based corpus from pickle data
class Corpus(object):

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        # character sets
        self.charset = self.generate_corpus()

        # character to index
        self.char2index = self.char_to_index()

        # index to character
        self.index2char = self.index_to_char()

    def generate_corpus(self):
        charset = set()
        for tweet in self.data['tweet']:
            charset = charset.union(set(tweet.lower()))
        # charset = charset.difference(MEANINGLESS_CHAR)
        return sorted(list(charset))

    def char_to_index(self):
        char_dict = dict((c, i+1) for i, c in enumerate(self.charset))
        return char_dict

    def index_to_char(self):
        char_dict = dict((i+1, c) for i, c in enumerate(self.charset))
        return char_dict

    def code_sentense(self, tweets):
        X = []
        for tweet in tweets:
            row = []
            for char in tweet:
                index = self.char2index.get(char)
                if index is None:
                    continue
                row.append(index)
            X.append(row)
        return X


if __name__ == '__main__':
    corpus = Corpus('data/sample_data.pkl')
    print("#"*20)
    print("number of charsets {}".format(len(corpus.charset)))
    print(corpus.charset)
    print(corpus.char2index['a'])