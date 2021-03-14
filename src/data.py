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


