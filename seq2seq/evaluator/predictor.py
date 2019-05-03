import torch

from seq2seq import IGNORE_INDEX


class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def predict(self, src_seq, attention_target=None):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        if attention_target is not None:
            attention_target = torch.LongTensor([IGNORE_INDEX] + [int(i) for i in attention_target]).view(1, -1)
            target_variables = dict(attention_target=attention_target)
        else:
            target_variables = None

        softmax_list, _, other = self.model(src_id_seq,
                                            input_lengths=[len(src_seq)],
                                            target_variables=target_variables)

        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]

        return tgt_seq, other
