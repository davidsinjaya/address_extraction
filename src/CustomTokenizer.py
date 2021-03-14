from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from keras.preprocessing.text import Tokenizer

# Limit on the number of features to K features.
TOP_K = 20000

# Limit on the length of text sequences.
# Sequences longer than this will be truncated.
# and less than it will be padded
MAX_SEQUENCE_LENGTH = 50


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

    def vectorize_input(self, tweets):
        # Vectorize training and validation texts.

        tweets = self.tokenizer.texts_to_sequences(tweets)
        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated
        # at the beginning.
        tweets = sequence.pad_sequences(tweets, maxlen=self.max_length, truncating='post', padding='post')
        return tweets