import torchtext

from .fields import SourceField, TargetField, AttentionField


def get_single_data(columns, tabular_data_fields):
    """Gets a dataset composed of a single example given `columns` which is a list of str representing the columns."""
    example = torchtext.data.Example.fromlist([col.split() for col in columns], tabular_data_fields)
    return torchtext.data.Dataset([example], tabular_data_fields)


def get_tabular_data_fields(is_predict_eos=True, is_attnloss=False, content_method="dot"):
    """Gets the data fields."""
    src = SourceField()
    tgt = TargetField(include_eos=is_predict_eos)

    tabular_data_fields = [('src', src), ('tgt', tgt)]

    if is_attnloss or content_method == 'hard':
        attn = AttentionField(use_vocab=False)
        tabular_data_fields.append(('attn', attn))

    return tabular_data_fields


def get_data(path, max_len, fields, format_ext='tsv'):
    """Gets the formated data."""
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    data = torchtext.data.TabularDataset(path=path, format=format_ext, fields=fields, filter_pred=len_filter)

    return data


def get_train_dev(train_path, dev_path, max_len, src_vocab, tgt_vocab, content_method,
                  is_predict_eos=True,
                  is_attnloss=False,
                  oneshot_path=None):
    """Get the fromatted train and dev data."""
    tabular_data_fields = get_tabular_data_fields(content_method=content_method,
                                                  is_predict_eos=is_predict_eos,
                                                  is_attnloss=is_attnloss)

    train = get_data(train_path, max_len, tabular_data_fields)
    dev = get_data(dev_path, max_len, tabular_data_fields)

    if oneshot_path is not None:
        oneshot = get_data(oneshot_path, max_len, tabular_data_fields)
    else:
        oneshot = None

    tabular_data_fields = dict(tabular_data_fields)
    src = tabular_data_fields["src"]
    tgt = tabular_data_fields["tgt"]

    if oneshot is not None:
        src.build_vocab(oneshot, max_size=src_vocab)
        tgt.build_vocab(oneshot, max_size=tgt_vocab)
    else:
        src.build_vocab(train, max_size=src_vocab)
        tgt.build_vocab(train, max_size=tgt_vocab)

    return train, dev, src, tgt, oneshot
