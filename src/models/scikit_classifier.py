import joblib
import numpy as np
import re
from multiprocessing import Pool


class ScikitClassifier:
    """
    Adapted from https://nlpforhackers.io/training-ner-large-dataset/
    """

    def __init__(self, word2vec=None, clf=None, search=None):
        self.word2vec = word2vec
        self.clf = clf
        self.search = search

    def shape(self, word):
        word_shape = 'other'
        if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
            word_shape = 'number'
        elif re.match('\W+$', word):
            word_shape = 'punct'
        elif re.match('[A-Z][a-z]+$', word):
            word_shape = 'capitalized'
        elif re.match('[A-Z]+$', word):
            word_shape = 'uppercase'
        elif re.match('[a-z]+$', word):
            word_shape = 'lowercase'
        elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
            word_shape = 'camelcase'
        elif re.match('[A-Za-z]+$', word):
            word_shape = 'mixedcase'
        elif re.match('__.+__$', word):
            word_shape = 'wildcard'
        elif re.match('[A-Za-z0-9]+\.$', word):
            word_shape = 'ending-dot'
        elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
            word_shape = 'abbreviation'
        elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
            word_shape = 'contains-hyphen'
        return word_shape

    def features(self, tokens, index, history):
        """
        `tokens`  = a POS-tagged sentence [(w1, t1), ...]
        `index`   = the index of the token we want to extract features for
        `history` = the previous IOB tag
        """

        # Pad the sequence with placeholders
        tokens = ['__START2__', '__START1__'] + tokens + ['__END1__', '__END2__']
        history = ['__START2__', '__START1__'] + history

        # shift the index with 2, to accommodate the padding
        index += 2

        word = tokens[index]
        prevword = tokens[index-1]
        prevprevword = tokens[index-2]
        nextword = tokens[index+1]
        nextnextword = tokens[index+2]
        previob = history[-1]
        prevpreviob = history[-2]

        feat_words = {
            'word': word,
            'next-word': nextword,
            # 'next-next-word': nextnextword,
            'prev-word': prevword,
            # 'prev-prev-word': prevprevword,
        }

        if self.word2vec is not None:
            new_feat = {}
            for col, word in feat_words.items():
                w2v = self.word2vec.wv
                size = self.word2vec.vector_size
                vec = w2v[word] if word in w2v else np.zeros((size))
                vec_dict = {f'{col}-vec-{i}': v for i, v in enumerate(vec)}
                new_feat = {**new_feat, **vec_dict}
            feat_words = new_feat

        feat_others = {
            'is_first': prevword == '__START1__',
            'is_last': nextword == '__END1__',
            'shape': self.shape(word),
            'next-shape': self.shape(nextword),
            # 'next-next-shape': self.shape(nextnextword),
            'prev-shape': self.shape(prevword),
            # 'prev-prev-shape': self.shape(prevprevword),
            'prev-iob': previob,
            # 'prev-prev-iob': prevpreviob,
        }

        feat_dict = {**feat_words, **feat_others}

        return feat_dict

    def transform(self, list_of_tagged):
        """
        Get X_train and y_train from list of tagged sentences
        :param list_of_tagged: [ [(word0, 'OTHER'), (word1, 'STREET'), ...], [...], ... ]
        :return X: [{...}, {...}, {...}, ...]
        :return y: ['OTHER', 'STREET', ...]
        """
        X, y = [], []
        for tagged in list_of_tagged:
            sentence = [w for w, t in tagged]
            history = []
            for idx in range(len(tagged)):
                label = tagged[idx][1]
                history.append(label)
                X.append(self.features(sentence, idx, history))
                y.append(label)
        self.X_train, self.y_train = X, y

    def fit(self):
        if self.search is not None:
            self.search.fit(self.X_train, self.y_train)
            self.clf = self.search.best_estimator_
        else:
            self.clf.fit(self.X_train, self.y_train)

    def predict(self, list_of_sentence):
        """
        Get y_pred from list of sentence
        :param list_of_sentence: [ [word0, word1, ...] , [word0, word1, ...], ...]
        :return: [ ['OTHER', 'STREET', ...], ['OTHER', 'POI', ...], ...]
        """
        list_of_preds = []
        for sentence in list_of_sentence:
            history = []
            for idx in range(len(sentence)):
                X = self.features(sentence, idx, history)
                y = self.clf.predict(X)[0]
                history.append(y)
            y_pred = [(w, t) for w, t in zip(sentence, history)]
            list_of_preds.append(y_pred)
        return list_of_preds

    def parallelize_predict(self, list_of_sentence, n_cores=10):
        list_split = np.split(list_of_sentence, n_cores)
        pool = Pool(n_cores)
        list_results = []
        for result in pool.map(self.predict, list_split):
            list_results += result
        pool.close()
        pool.join()
        return list_results

    def save_model(self, save_path):
        joblib.dump(self.clf, save_path)

    def load_model(self, load_path):
        self.clf = joblib.load(load_path)