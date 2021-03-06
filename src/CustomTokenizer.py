from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Limit on the number of features to K features.
TOP_K = 20000

# Limit on the length of text sequences.
# Sequences longer than this will be truncated.
# and less than it will be padded
MAX_SEQUENCE_LENGTH = 30

def pad_sequence_tokens(tokens_onehot):
    return pad_sequences(tokens_onehot, maxlen=MAX_SEQUENCE_LENGTH, truncating='post', padding='post', value=[0,0,1])

class CustomTokenizer:
    def __init__(self, train_texts):
        self.train_texts = train_texts
        self.tokenizer = Tokenizer(num_words=TOP_K)

    def train_tokenize(self):
        # Get max sequence length.
        max_length = len(max(self.train_texts, key=len))
        self.max_length = min(max_length, MAX_SEQUENCE_LENGTH)

        # Create vocabulary with training texts.
        self.tokenizer.fit_on_texts(self.train_texts)

    def vectorize_input(self, input):
        input = self.tokenizer.texts_to_sequences(input)
        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated
        # at the beginning.
        output = sequence.pad_sequences(input, maxlen=self.max_length, truncating='post', padding='post')
        return output

    def reverse_tokenized_to_array_text(self, input):
        reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))
        output = []
        for tokens in input:
            output.append([reverse_word_map[token] if token != 0 else None for token in tokens])
        return output


