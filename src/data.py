import numpy as np

def raw_to_tokens(raw):
    return raw.lower().split(' ')


def tokens_to_tagged(raw, poi, street):
    """
    Convert tokenized sentence to tagged sentence based on provide POI and STREET
    :param raw: e.g. [word0, word1, word2, word3, word4]
    :param poi: e.g. [word0, word1]
    :param street: e.g. [word3, word4]
    :return: e.g. [(word0, 'POI'), (word1, 'POI'), (word2, 'OTHER'), (word3, 'STREET'), (word4, 'STREET')]
    """
    tagged = []
    for word in raw:
        if word in poi:
            tagged.append((word, 'POI'))
        elif word in street:
            tagged.append((word, 'STREET'))
        else:
            tagged.append((word, 'OTHER'))
    return tagged


def tagged_to_tokens(tagged):
    """
    Convert tagged sentence back to POI and STREET
    :param tagged: e.g. [(word0, 'POI'), (word1, 'POI'), (word2, 'OTHER'), (word3, 'STREET'), (word4, 'STREET')]
    :return raw: e.g. [word0, word1, word2, word3, word4]
    :return poi: e.g. [word0, word1]
    :return street: e.g. [word3, word4]
    """
    raw = []
    poi = []
    street = []
    for word, tag in tagged:
        raw.append(word)
        if tag == 'POI':
            poi.append(word)
        if tag == 'STREET':
            street.append(word)
    return raw, poi, street


def tagged_to_poi_street(tagged):
    """
    Convert tagged sentence back to POI/street
    :param tagged: e.g. [(word0, 'POI'), (word1, 'POI'), (word2, 'OTHER'), (word3, 'STREET'), (word4, 'STREET')]
    :return: e.g. [ word0 word1/word3 word4]
    """
    raw, poi, street = tagged_to_tokens(tagged)
    return ' '.join(poi) + '/' + ' '.join(street)


def tagged_tokens_to_onehot(tagged_tokens):
    """
    Convert tagged sentence to POI/street one-hot encoding (3D). Format [POI, STREET, OTHER]
    :param tagged: e.g. [(word0, 'POI'), (word1, 'POI'), (word2, 'OTHER'), (word3, 'STREET'), (word4, 'STREET')]
    :return tokens_onehot: e.g. [[1,0,0], [1,0,0], [0,0,1], [0,1,0], [0,1,0]]
    """
    onehot=[]
    for pair in tagged_tokens:
        tag = pair[1] # get the token type
        if tag == 'POI':
            res=[1,0,0]
        elif tag == 'STREET':
            res=[0,1,0]
        else:
            res=[0,0,1] #tag others
        onehot.append(res)
    return np.array(onehot)


def get_predicted_tokens(tokens_onehot, raw_tokens):
    """
   Get the predicted POI/street by mapping onehot encoder to raw tokens.
   :param tokens_onehot: e.g. [[1,0,0], [1,0,0], [0,0,1], [0,1,0], [0,1,0]]
   :param raw_tokens: e.g. ['word0', 'word1', word2', 'word3', 'word4']
   :return POI/street: e.g. [word0 word1/word3 word4]
   """
    pred = np.array(tokens_onehot)
    ix_poi = np.where(pred == 0)[0]
    ix_street = np.where(pred == 1)[0]

    pred_poi = []
    pred_street = []
    res = ''
    if len(ix_poi) > 0:
        for ix in ix_poi:
            if raw_tokens[ix] is not None:  # omit tagging on padding
                pred_poi.append(raw_tokens[ix])
        res += ' '.join(pred_poi)

    res += '/'
    if len(ix_street) > 0:
        for ix in ix_street:
            if raw_tokens[ix] is not None:  # omit tagging on padding
                pred_street.append(raw_tokens[ix])
        res += ' '.join(pred_street)

    return res


