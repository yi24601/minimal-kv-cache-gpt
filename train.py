class CharEncoder:
    def __init__(self,all_chars):
        
        self.char2index = dict()
        self.index2char = dict()
        all_chars = sorted(all_chars)
        all_chars = sorted(all_chars)

        for index,char in enumerate(all_chars):
            self.char2index[char] = index
            self.index2char[index] = char

    def encode(self, sentence):
        return [self.char2index[char] for char in sentence]

    def decode(self, index_list):
        return ''.join([self.index2char[index] for index in index_list])