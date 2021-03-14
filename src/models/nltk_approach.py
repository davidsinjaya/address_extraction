from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline


class NltkApproach:
    """
    Adapted from https://nlpforhackers.io/training-pos-tagger/
    """

    def _untag(self, tagged):
        """
        Returns
        :param tagged: [(word0, 'OTHER'), (word1, 'STREET'), (word2, 'STREET'), ...]
        :return: [word0, word1, word2, ...]
        """
        return [w for w, t in tagged]

    def _features(self, sentence, index):
        """
        :param sentence: [word0, word1, word2, ...]
        :param index: the index of the word
        :return: dictionary of features for the word
        """
        return {
            'word': sentence[index],
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': False if len(sentence[index]) == 0 else sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            'is_all_lower': sentence[index].lower() == sentence[index],
            # 'prefix-1': sentence[index][0],
            # 'prefix-2': sentence[index][:2],
            # 'prefix-3': sentence[index][:3],
            # 'suffix-1': sentence[index][-1],
            # 'suffix-2': sentence[index][-2:],
            # 'suffix-3': sentence[index][-3:],
            'prev_word': '' if index == 0 else sentence[index - 1],
            'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
            'has_hyphen': '-' in sentence[index],
            'is_numeric': sentence[index].isdigit(),
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
        }

    def get_features(self, sentence):
        """
        Convert words in sentence to features
        :param sentence: [word0, word1, word2, ...]
        :return: [{...}, {...}, {...}, ...]
        """
        return [self._features(sentence, idx) for idx in range(len(sentence))]

    def get_label(self, tagged):
        """
        Get label from tagged sentence
        :param tagged: [(word0, 'OTHER'), (word1, 'STREET'), (word2, 'STREET'), ...]
        :return: ['OTHER', 'STREET', 'STREET']
        """
        return [t for w, t in tagged]

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
            X += self.get_features(sentence)
            y += self.get_label(tagged)
        self.X_train, self.y_train = X, y

    def fit(self):
        self.clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=True)),
            ('classifier', DecisionTreeClassifier(criterion='entropy'))
        ])
        self.clf.fit(self.X_train, self.y_train)

    def predict(self, list_of_sentence):
        """
        Get y_pred from list of sentence
        :param list_of_sentence: [ [word0, word1, ...] , [word0, word1, ...], ...]
        :return: [ ['OTHER', 'STREET', ...], ['OTHER', 'POI', ...], ...]
        """
        list_of_preds = []
        for sentence in list_of_sentence:
            X = self.get_features(sentence)
            y = self.clf.predict(X)
            y_pred = [(w, t) for w, t in zip(sentence, y)]
            list_of_preds.append(y_pred)
        return list_of_preds




