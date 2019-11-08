import os
import numpy as np
from pathlib import Path
from functools import partial

TRAIN_SPLIT_PCT = .95
VAL_SPLIT_PCT = .025
DEFAULT_STROKE_LENGTH = 750
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHAR_LENGTH = 30
# remove uncommon characters
REMOVE_CHARS = '+/0123456789:();#!'


def get_data_dir():
    data_dir = os.environ.get('HANDWRITING_GENERATION_DATA_DIR')
    if data_dir is None:
        return Path(os.path.abspath(__file__)).parent / '../data_2'
    return Path(data_dir)


DATA_DIR = get_data_dir()


def get_model_dir():
    d = os.environ.get('HANDWRITING_GENERATION_MODEL_DIR', './models')
    assert os.path.exists(d), ('Specify path to models as an env var: '
                               '`HANDWRITING_GENERATION_MODEL_DIR`')
    return Path(d)


def load_raw_data():
    """
    Load raw data.
    Mean stroke length is 644, max is 1191. 75% of strokes < 750 length.
    :return: sentences, strokes
    """
    with (DATA_DIR / 'sentences.txt').open('r') as f:
        sentences = f.readlines()
    strokes = np.load(DATA_DIR / 'strokes-py3.npy', allow_pickle=True)
    assert len(strokes) == 6000
    assert len(sentences) == 6000
    return sentences, strokes


def _tokenize(sentences):
    """
    Tokenize sentences into characters. There are 77 characters in total.
    Median sentence length is 29 characters, max of 64.
    :return: sentences_tok list(list(char)), char_set
    """
    sentences_tok = [list(s.strip()) for s in sentences]
    char_set = set([s for st in sentences_tok for s in st])
    char_set = char_set - set(REMOVE_CHARS)
    assert len(char_set) == 59
    return sentences_tok, char_set


def sent_to_int(sentence, char_dict):
    return np.array([char_dict.get(s, 1) for s in sentence])


def _tokenized_sentences_to_ints(sentences_tok, char_set):
    """
    Convert characters to ints for character generation
    """
    char_dict = {}
    char_dict[' '] = 0
    char_dict['<UNK>'] = 1
    idx = 1
    for c in char_set:
        if c == ' ':
            continue
        idx += 1
        char_dict[c] = idx

    sentences_tok = [sent_to_int(st, char_dict)
                     for st in sentences_tok]
    return sentences_tok, char_dict


def _preprocess_raw(sentences, strokes, sentences_to_int):
    """
    Preprocess raw data. Tokenize sentences
    :return: (tuple), dict
    """
    metadata = {}
    sentences_tok, char_set = _tokenize(sentences)

    if sentences_to_int:
        # convert each character to an integer
        sentences_tok, char_dict = _tokenized_sentences_to_ints(
            sentences_tok, char_set)
        metadata['char_dict'] = char_dict

    metadata['char_set'] = char_set
    metadata['vocab_size'] = len(char_dict)

    return (sentences_tok, strokes), metadata


def _split_data(sentences, strokes, seed=42):
    """
    Split into train/val/test splits
    :return: train, val, test data
    """
    np.random.seed(seed)

    # compute shuffle
    data_len = len(sentences)
    idxs = list(range(data_len))
    np.random.shuffle(idxs)

    # get split index
    train_idx = int(data_len * TRAIN_SPLIT_PCT)
    val_idx = int(data_len * (TRAIN_SPLIT_PCT + VAL_SPLIT_PCT))

    # get split index lists
    train_idxs = idxs[:train_idx]
    val_idxs = idxs[train_idx:val_idx]
    test_idxs = idxs[val_idx:]

    # compile train/val/test data tuples
    train_data = ([sentences[t] for t in train_idxs], strokes[train_idxs])
    val_data = ([sentences[t] for t in val_idxs], strokes[val_idxs])
    test_data = ([sentences[t] for t in test_idxs], strokes[test_idxs])

    return train_data, val_data, test_data


def get_preprocessed_data_splits(seed=42, sentences_to_int=True):
    """
    Load raw data, preprocess it, and then split the data into train/val/test
    :param seed: seed for random split of data
    :param sentences_to_int: bool, convert chars to ints in sentences
    :return: train, val, test data
    """
    sentences, strokes = load_raw_data()
    data, metadata = _preprocess_raw(sentences, strokes, sentences_to_int)
    train, val, test = _split_data(*data, seed)
    return train, val, test, metadata


def pad_vec(vecs, length, rand_start=False):
    """
    Zero pad `vecs` to `length`.
    truncate vecs longer than length, otherwise pad the end with zeros
    """
    padded_vecs = []
    for s in vecs:
        if s.shape[0] > length:
            max_start_idx = s.shape[0] - length

            if rand_start:
                start_idx = np.random.randint(0, max_start_idx)
            else:
                start_idx = 0

            end_idx = start_idx + length
            padded = s[start_idx:end_idx]
        else:
            zero_padding_len = max(length - s.shape[0], 0)

            pad_arg = (0, zero_padding_len)

            if len(s.shape) == 2:
                # don't pad extra dimension
                pad_arg = (pad_arg, (0, 0))

            padded = np.pad(s, pad_arg, mode='constant')

        assert padded.shape[0] == length
        padded_vecs.append(padded)

    return np.array(padded_vecs)


def _normalize(vec):
    # return (vec - np.mean(vec)) / np.std(vec)
    # we want the mean bias, so that the characters go from left to right
    # but the scale should be similar
    return vec / np.std(vec)


def _normalize_strokes(strokes, xstd, ymean, ystd):
    # normalize the x, y coordinates
    # but keep bias in x (to move left to right)
    reshape = strokes.reshape((-1, 3))
    reshape[:, 1] = reshape[:, 1] / xstd
    reshape[:, 2] = (reshape[:, 2] - ymean) / ystd
    return reshape.reshape(strokes.shape)


def batch_generator(data, length, tuple_idx,
                    batch_size=DEFAULT_BATCH_SIZE, normalize_strokes=False):
    """
    Iterate infinitely over batches on data for the unconditional generator.
    """
    data = data[tuple_idx]
    assert length > 2

    if normalize_strokes:
        # get the mean and std of dataset for x, y coordinates
        p = pad_vec(data, length + 1, rand_start=True).reshape((-1, 3))
        xstd = np.std(p[:, 1])
        ymean = np.mean(p[:, 2])
        ystd = np.std(p[:, 2])

    data_len = len(data)
    assert data_len > batch_size, "Data length is smaller than batch size"
    while True:
        idxs = np.random.choice(data_len, size=batch_size)
        batch_data = [data[i] for i in idxs]
        padded = pad_vec(batch_data, length + 1, rand_start=True)

        if normalize_strokes:
            padded = _normalize_strokes(padded, xstd, ymean, ystd)

        batch = np.array(padded)
        yield batch[:, :-1], batch[:, 1:]


character_batch_generator = partial(
    batch_generator, length=DEFAULT_CHAR_LENGTH, tuple_idx=0)

stroke_batch_generator = partial(
    batch_generator, length=DEFAULT_STROKE_LENGTH, tuple_idx=1,
    normalize_strokes=True)


def conditional_batch_generator(data, stroke_length=DEFAULT_STROKE_LENGTH,
                                char_length=DEFAULT_CHAR_LENGTH,
                                batch_size=DEFAULT_BATCH_SIZE,
                                normalize_strokes=True):
    """
    Generate data for conditional stroke generation - pad the
    strokes and characters along with random shuffle batches

    :param normalize_strokes: bool, whether to center and rescale
        the y-coordinates and rescale the x-coordinates
    """
    strokes = data[1]
    characters = data[0]

    if normalize_strokes:
        # get the mean and std of dataset for x, y coordinates
        p = pad_vec(strokes, stroke_length + 1).reshape((-1, 3))
        xstd = np.std(p[:, 1])
        ymean = np.mean(p[:, 2])
        ystd = np.std(p[:, 2])

    data_len = len(strokes)
    assert data_len > batch_size, "Data length is smaller than batch size"

    while True:
        idxs = np.random.choice(data_len, size=batch_size)
        batch_strokes = [strokes[i] for i in idxs]
        batch_chars = [characters[i] for i in idxs]

        padded_strokes = pad_vec(batch_strokes, stroke_length + 1)
        padded_chars = pad_vec(batch_chars, char_length)

        if normalize_strokes:
            padded_strokes = _normalize_strokes(
                padded_strokes, xstd, ymean, ystd)

        batch_strokes = np.array(padded_strokes)
        batch_chars = np.array(padded_chars)

        yield batch_strokes[:, :-1], batch_strokes[:, 1:], batch_chars
