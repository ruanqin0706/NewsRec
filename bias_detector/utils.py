import ast
import six
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import shutil
import os


# ref from keras_preprocessing/sequence.py
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# def load_elmo(path, max_len=200):
#     '''
#     load ELMo embedding from tsv file.
#     :param path: tsv file path.
#     :param to_pickle: Convert elmo embeddings to .npy file, avoid read and pad every time.
#     :return: elmo embedding and its label.
#     '''
#     X = []
#     label = []
#     l_encoder = LabelEncoder()
#     with open(path, 'rb') as inf:
#         for line in inf:
#             gzip_fields = line.decode('utf-8').split('\t')
#             gzip_label = gzip_fields[1]
#             elmo_embd_str = gzip_fields[4].strip()
#             elmo_embd_list = ast.literal_eval(elmo_embd_str)
#             elmo_embd_array = np.array(elmo_embd_list)
#             padded_seq = sequence.pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
#             X.append(padded_seq)
#             label.append(gzip_label)
#     Y = l_encoder.fit_transform(label)
#
#     return np.array(X), np.array(Y)


def load_elmo_list(path_list, category_size=-1, max_len=200, shuffle=False):
    X = []
    label = []
    l_encoder = LabelEncoder()
    for path in path_list:
        with open(path, 'rb') as inf:
            for i, line in enumerate(inf):
                if category_size == -1 or i+1 <= category_size:
                    gzip_fields = line.decode('utf-8').split('\t')
                    gzip_label = gzip_fields[1]
                    elmo_embd_str = gzip_fields[4].strip()
                    elmo_embd_list = ast.literal_eval(elmo_embd_str)
                    elmo_embd_array = np.array(elmo_embd_list)
                    padded_seq = pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
                    X.append(padded_seq)
                    label.append(gzip_label)
                else:
                    break
    Y = l_encoder.fit_transform(label)
    X, Y = np.array(X), np.array(Y)
    if shuffle:
        X, Y = unison_shuffled_copies(a=X, b=Y)
    return X, Y


def load_elmo_list2(path_list, category_size, max_len=200):
    l_encoder = LabelEncoder()
    zero_X = []
    zero_label = []
    one_X = []
    one_label = []
    for path in path_list:
        with open(path, 'rb') as inf:
            for i, line in enumerate(inf):
                if len(zero_label) >= category_size and len(one_label) >= category_size:
                    break

                gzip_fields = line.decode('utf-8').split('\t')
                gzip_label = gzip_fields[1]
                elmo_embd_str = gzip_fields[4].strip()
                elmo_embd_list = ast.literal_eval(elmo_embd_str)
                elmo_embd_array = np.array(elmo_embd_list)
                padded_seq = pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]

                if gzip_label == "true":
                    one_X.append(padded_seq)
                    one_label.append(gzip_label)
                else:
                    zero_X.append(padded_seq)
                    zero_label.append(gzip_label)

        if len(zero_label) >= category_size and len(one_label) >= category_size:
            break
    X = zero_X[:category_size] + one_X[:category_size]
    label = zero_label[:category_size] + one_label[:category_size]
    assert len(X) == len(label)
    del zero_X, one_X, zero_label, one_label
    Y = l_encoder.fit_transform(label)
    X, Y = np.array(X), np.array(Y)
    X, Y = unison_shuffled_copies(a=X, b=Y)
    return X, Y


def create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def collate_func(data_batch, device_no=-1):
    features, labels = zip(*data_batch)
    features = np.asarray(features)
    features = torch.tensor(features)
    labels = torch.tensor(labels)
    features = features.permute(0, 2, 1)
    labels = labels.float()

    if device_no > -1:
        features = features.cuda(device_no)
        labels = labels.cuda(device_no)

    return features, labels


def cal_threshold(strategy='static01', thr_start=0.0, thr_end=1.0, step_rate=0):
    if strategy.startswith('static'):
        return thr_start, thr_end
    elif strategy == 'tsa_exp':
        alpha = np.exp((step_rate-1)*5)
        thr = alpha * 0.5 + 0.5
        return 1 - thr, thr
    elif strategy == 'tsa_log':
        alpha = 1-np.exp(-step_rate*5)
        thr = alpha * 0.5 + 0.5
        return 1 - thr, thr
    elif strategy == 'tsa_linear':
        alpha = step_rate
        thr = alpha * 0.5 + 0.5
        return 1 - thr, thr
