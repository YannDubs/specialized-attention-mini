"""
Main package entrypoint.
"""

import os
import logging
import math
import json
import shutil

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss.loss import get_losses, LossWeightUpdater
from seq2seq.metrics.metrics import get_metrics
from seq2seq.dataset.helpers import get_train_dev
from seq2seq.util.callbacks import EarlyStopping
from seq2seq.util.helpers import Rate2Steps, regularization_loss, get_latest_file
from seq2seq.util.torchextend import MLP
from seq2seq.attention import (Attender, LocationOnlyAttender, ContentOnlyAttender,
                               HardAttender)
from seq2seq.attention.location import get_regularizers_location

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG,)


def get_attender(name, attender_kwargs):
    if name == "attender":
        return dict(attender=Attender, attender_kwargs=attender_kwargs)
    elif name == "location":
        return dict(attender=LocationOnlyAttender,
                    attender_kwargs=attender_kwargs["location_kwargs"])
    elif name == "content":
        return dict(attender=ContentOnlyAttender,
                    attender_kwargs=attender_kwargs["content_kwargs"])
    elif name == "hard":
        return dict(attender=HardAttender, attender_kwargs={})
    else:
        raise ValueError("Unkown name={}.".format(name))


def _save_parameters(args, directory, filename="train_arguments.txt"):
    """Save arguments to a file given a dictionary."""
    with open(os.path.join(directory, filename), 'w') as f:
        json.dump(args, f, indent=4, sort_keys=True)


def _rename_latest_file(path, new_name):
    """Rename the latest modified/added file in a path."""
    latest_file = get_latest_file(path)
    os.rename(latest_file, os.path.join(path, new_name))


def get_is_attn_field(attender, loss_names, force_mu):
    """Whether to add the attention field."""
    is_hard_attn = attender == "hard"
    is_attn_loss = False
    for loss_name in loss_names:
        if isinstance(loss_name, str) and "attention" in loss_names:
            is_attn_loss = True
        elif "attention" in loss_name[0]:
            is_attn_loss = True
    return is_hard_attn or is_attn_loss or force_mu is not None


def get_seq2seq_model(src_len,
                      tgt_len,
                      max_len,
                      tgt_sos_id,
                      tgt_eos_id,
                      total_training_calls,
                      rnn_cell="gru",
                      is_mlps=False,
                      embedding_size=64,
                      hidden_size=128,
                      mid_dropout=0.5,
                      gating_encoder="residual",  # DEV MODE
                      content_method='scaledot',  # see if scaledmult better
                      value_size=-1,
                      value_noise_sigma=0,
                      n_steps_prepare_pos=100,
                      positioning_method="gaussian",
                      rate_start_round=0.05,
                      anneal_temp_round=0.1,
                      rounder_mu="concrete",
                      mode_attn_mix="loc_conf",  # TO DO - medium: chose best and remove parameter
                      rate_attmix_wait=0,
                      dflt_perc_loc=0.5,  # TO DO - medium: chose best and remove parameter
                      rounder_perc="concrete",  # TO DO - medium: chose best and remove parameter
                      is_dev_mode=False,
                      is_viz_train=False,
                      attender="attender",
                      is_reg_clamp_mu=True,  # DEV MODE
                      pretrained_locator=None,  # DEV MODE
                      gating="highway",  # DEV MODE
                      is_diagonal=False,  # DEV MODE
                      clipping_step=3,  # DEV MODE
                      is_rnn_loc=True,  # DEV MODE
                      is_l0=False,  # DEV MODE
                      is_reg_mu_gates=False,  # DEV MODE
                      location_size=64,  # DEV MODE
                      rounder_weights=None,  # DEV MODE
                      is_force_sigma=False,  # DEV MODE
                      force_mu=None,  # DEV MODE
                      ):
    """Return a initialized extrapolator model.

    Args:
        src_len (int): size of source vocab.
        tgt_len (int): size of target vocab.
        max_len (int): maximum possible length of any source sentence.
        total_training_calls (int): number of maximum training calls.
        is_mlps (bool, optional): whether to use MLPs for the generators instead
            of a linear layer.
        embedding_size (int, optional): size of embedding for the decoder and
            encoder.
        hidden_size (int, optional): hidden size for unidirectional encoder.
        mid_dropout (float, optional): dropout between
            the decoder and encoder.
        content_method ({'multiplicative', "additive", "euclidean", "scaledot",
            "cosine", "kq"}, optional):
            The method to compute the alignment. `"scaledot" [Vaswani et al., 2017]
            mitigates the high dimensional issue by rescaling the dot product.
            `"additive"` is the original  attention [Bahdanau et al., 2015].
            `"multiplicative"` is faster and more space efficient [Luong et al., 2015]
            but performs a little bit worst for high dimensions. `"cosine"` cosine
            distance. `"euclidean"` Euclidean distance. "kq" first uses 2 different
            mlps to convert the encoder and decoder hidden state to low dimensional
            key and queries.
        values_size (int, optional): size of the generated value. -1 means same
            as hidden size. Can also give percentage of hidden size betwen 0 and 1.
        value_noise_sigma (float, optional): relative noise to add to
            the output of the key and query generator. This can be seen as a rough
            approximation to building a variational latent representation as it
            forces the prediction of a distribution rather than points.
        n_steps_prepare_pos (int, optional): number steps during which
            to consider the positioning as in a preparation mode. During
            preparation mode, the model have less parameters to tweak, it will
            focus on what I thought were the most crucial bits. For example it
            will have a fix sigma and won't have many of the regularization term,
            this is to help it start at a decent place in a lower dimensional
            space, before going to the hard task of tweaking all at the same time.
        positioning_method ({"gaussian", "laplace"}, optional): name of the
            positional distribution. `laplace` is more human plausible but
            `gaussian` works best.
        is_posrnn (bool, optional): whether to use a rnn for the positional
            attention generator.
        is_building_blocks_mu (bool, optional): whether to use building blocks to
            generate the positional mu rather than using a normal MLP.
        is_bb_bias (bool, optional): adding a bias term to the building blocks.
            THis has the advantage of letting the network go to absolut positions
            (ex: middle, end, ...). THe disadvantage being that the model will often
            stop using other building blocks and thus be less general.
        rate_start_rounding (float, optional): percentage of training steps to
            wait before starting the rounding of all variables to round.
        anneal_temp_round (float, optional): percentage of training steps for
            which to anneal the temperature in the rounding of all variables to round.
        rounder_mu({“concrete”, “stochastic”, None}, optional): the method for
            approximative differential rounding to use for rounding mu to the
            position of words. If `concrete` it will use the concrete distribution
            similarly to “The Concrete Distribution: A Continuous Relaxation of
            Discrete Random Variables”. If `stochastic` it will round to the
            ceiling with a probability equal to the decimal of x and then use
            straight through estimators in the backward pass. If `None` will not
            use any rounding. Rounding is desirable to make the position attention
            look at the correct position even for sentences longer than it have
            ever seen.
        mode ({"generated","normalized_pos_conf","pos_conf"}, optional) how to
            generate the position confidence when mixing the content and positioning
            attention. `generated` will generate one from the controller,
            this might give good results but is less interpretable. `mean_conf`
            will normalize the positional confidence by `(position_confidence
            + content_confidence)`, this will force meaningfull confidences for
            both attentions. The latter should not be used when not using sequential
            attention because pos% will always be 0.5 if both are confident, i.e
            content cannot just be used for position to help it.`pos_conf` will
            directly use the position cofidence, this will force meaningfull
            positioning confidence but not the content ones. This also says
            to the network that if position is confident use it regardless of content
            because it's more extrapolable.
        rate_attnmix_wait (float, optional): percentage of training steps to wait
            for before starting to generate the positional percentage. Until then
            will use `dflt_perc_loc`.
        dflt_perc_loc (float, optional): constant positional percentage to
            use while `rate_attnmix_wait`.
        rounder_perc ({“concrete”, “stochastic”, None}, optional): the method
            for approximative differential rounding to use for rounding mu to the
            position of words. If `concrete` it will use the concrete distribution
            similarly to “The Concrete Distribution: A Continuous Relaxation of
            Discrete Random Variables”. If `stochastic` it will round to the ceiling
            with a probability equal to the decimal of x and then use straight
            through estimators in the backward pass. If `None` will not use any
            rounding. Rounding is desirable to make the position attention look
            at the correct position even for sentences longer than it have ever seen.
        is_dev_mode (bool, optional): whether to store many useful variables in
            `additional`. Useful when predicting with a trained model in dev mode
             to understand what the model is doing. Use with `dev_predict`.
        is_viz_train (bool, optional): whether to save how the averages of some
            intepretable variables change during training in "visualization"
            of `additional`.
        force_mu ({None, "bb", "bb-zeroW", "all"}, optional): If `"bb"` add a
            provided attention as building block, it still has to learn the
            correct weights. If `"bb-zeroW"` it zeros out the weights of all othet
            building blocks in addition to adding the provided attention as building
            block, the network still has to learn to use a weight of 1 for this bb.
            `"all"` sets both the correct building block and weights, the network
            will have access to the correct mu but still has to learn sigma.  `
            None` doesn't force mu.
    """
    assert max_len > 1, "Max len has to be greater than 1"

    # interpolating rates to interpolating steps
    rate2steps = Rate2Steps(total_training_calls)

    Generator = MLP if is_mlps else nn.Linear

    # Encoder
    value_kwargs = dict(output_size=value_size,
                        Generator=Generator,
                        gating=gating_encoder)

    encoder = EncoderRNN(src_len,
                         max_len,
                         hidden_size,
                         embedding_size,
                         value_kwargs=value_kwargs,
                         rnn_cell=rnn_cell)

    # Decoder
    n_steps_start_round = rate2steps(rate_start_round)
    rounders_kwargs = {"concrete": {"name": "concrete",
                                    "n_steps_interpolate": rate2steps(anneal_temp_round),
                                    "start_step": n_steps_start_round},
                       "softConcrete": {"name": "softConcrete",
                                        "n_steps_interpolate": rate2steps(anneal_temp_round),
                                        "start_step": n_steps_start_round},
                       "stochastic": {"name": "stochastic",
                                      "start_step": n_steps_start_round},
                       None: {"name": None},
                       "plateau": {"name": "plateau"}}

    sigma_kwargs = dict(is_force_sigma=is_force_sigma)

    mu_kwargs = dict(rounder_mu_kwargs=rounders_kwargs[rounder_mu],
                     is_reg_clamp_mu=is_reg_clamp_mu,
                     is_diagonal=is_diagonal,
                     clipping_step=clipping_step,
                     is_l0=is_l0,
                     is_reg_mu_gates=is_reg_mu_gates,
                     rounder_weights_kwargs=rounders_kwargs[rounder_weights],
                     force_mu=force_mu)

    location_kwargs = dict(n_steps_prepare_pos=n_steps_prepare_pos,
                           pdf=positioning_method,
                           Generator=Generator,
                           mu_kwargs=mu_kwargs,
                           pretrained_locator=pretrained_locator,
                           gating=gating,
                           is_recurrent=is_rnn_loc,
                           hidden_size=location_size,
                           sigma_kwargs=sigma_kwargs)

    content_kwargs = dict(scorer=content_method)

    n_steps_wait = rate2steps(rate_attmix_wait)
    mixer_kwargs = dict(Generator=Generator,
                        mode=mode_attn_mix,
                        n_steps_wait=n_steps_wait,
                        rounder_perc_kwargs=rounders_kwargs[rounder_perc],
                        dflt_perc_loc=dflt_perc_loc)

    attender_kwargs = dict(content_kwargs=content_kwargs,
                           location_kwargs=location_kwargs,
                           mixer_kwargs=mixer_kwargs)

    decoder = DecoderRNN(tgt_len,
                         max_len,
                         hidden_size,
                         embedding_size,
                         tgt_sos_id,
                         tgt_eos_id,
                         value_size=encoder.value_size,
                         rnn_cell=rnn_cell,
                         **get_attender(attender, attender_kwargs))

    seq2seq = Seq2seq(encoder, decoder, mid_dropout=mid_dropout)

    seq2seq.set_dev_mode(value=is_dev_mode)
    seq2seq.set_viz_train(value=is_viz_train)

    logger.debug(str(seq2seq))

    return seq2seq


def train(train_path,
          dev_path,
          metric_names=["word accuracy", "sequence accuracy",
                        "final target accuracy"],
          loss_names=["nll"],
          max_len=50,
          epochs=100,
          output_dir="models/",
          src_vocab=50000,
          tgt_vocab=50000,
          is_predict_eos=True,
          teacher_forcing=0.2,
          batch_size=32,
          eval_batch_size=256,
          lr=1e-3,
          save_every=100,
          print_every=100,
          log_level="info",
          cuda_device=0,
          optim=None,
          grad_clip_value=3,
          grad_clip_norm=5,
          resume=False,
          checkpoint_path=None,
          patience=15,
          name_checkpoint=None,
          eos_weight=1,
          anneal_eos_weight=0,  # TO DO : hyperparmeter optimize
          _initial_eos_weight=0.,
          content_method='scaledot',
          is_amsgrad=False,  # TO DO - medium : chose best valeu and delete param
          rate_prepare_pos=0.05,
          plateau_reduce_lr=[4, 0.5],
          _initial_model="initial_model",
          attender="attender",
          pretrained_locator=None,  # DEV MODE
          force_mu=None,  # DEV MODE
          **kwargs):
    """Trains the model given all parameters.

    Args:
        train_path (str): path to the training data.
        dev_path (str): path to the validation data.
        metric_names (list of str, optional): names of the metrics to use. See
            `seq2seq.metrics.metrics.get_metrics` for more details.
        loss_names (list of str, optional): names of the metrics to use. See
            `seq2seq.loss.loss.get_losses` for more details.
        max_len (int, optional): maximum possible length of any source sentence.
        epochs (int, optional): maximum number of training epochs.
        output_dir (str, optional): path to the directory where the model
            checkpoints should be stored.
        src_vocab (int, optional): maximum source vocabulary size.
        tgt_vocab (int, optional): maximum target vocabulary size.
        is_predict_eos (bool, optional): whether the mdoel has to predict the <eos>
            token.
        teacher_forcing (float, optional): teacher forcing ratio.
        batch_size (int, optional): size of each training batch.
        eval_batch_size (int, optional): size of each evaluation batch.
        lr (float, optional): learning rate.
        save_every (int, optional): Every how many batches the model should be saved.
        print_every (int, optional): Every how many batches to print results.
        log_level (str, optional): Logging level.
        cuda_device (int, optional): Set cuda device to use .
        optim ({'adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'}, optional):
            Name of the optimizer to use.
        grad_clip_norm (float, optional): L2 Norm to which to clip the gradients.
            Good default: 1. (default: 5)
        grad_clip_value (float, optional): Values to which to clip the gradients.
            Good default: 0.5. (default: 2)
        resume (bool, optional): Whether to resume training from the latest checkpoint.
        checkpoint_path (str, optional): path to load checkpoint from in case
            training should be resumed
        patience (int, optional): patience if using early stoping. If `None`
            doesn't use any early stoping.
        name_checkpoint (str, optional): name to give to the checkpoint to make
            it more human readable.
        eos_weight (int, optional): weight of the loss that should be given to
            the <eos> token.
        anneal_eos_weight (float, optional): if not 0 , it will force
            the network to keep a lower eos weight while it's learning. eos_weight
            will actually start at `_initial_eos_weight` and will linearly
            increase or decrease (if `_initial_eos_weight` > `eos_weight`)
            at each training calls, until it reaches the given `eos_weight`. This
            parameter defines the percentage of training calls before the mdoel
            should reach the final `eos_weight`.
        attention ({"post-rnn", "pre-rnn", None}, optional): where to use attention.
        content_method({'multiplicative', "additive, "scaledot", "dot", "hard"}, optional):
            The method to compute the alignment. `"dot"` corresponds to a simple
            product. `"additive"` is the original  attention [Bahdanau et al., 2015].
            `"Multiplicative"` is faster and more space efficient [Luong et al., 2015]
            but performs a little bit worst for high dimensions. `"scaledot"
            [Vaswani et al., 2017] mitigates the highdimensional issue by rescaling
            the dot product. `"scalemult"` uses the same rescaling trick but with
            a multiplicative attention.
        is_amsgrad (bool, optional): Whether to use amsgrad, which is supposed
            to make Adam more stable : "On the Convergence of Adam and Beyond".
        rate_prepare_pos (int, optional): percentage of total steps during which
            to consider the positioning as in a preparation mode. During
            preparation mode, the model have less parameters to tweak, it will
            focus on what I thought were the most crucial bits. For example it
            will have a fix sigma and won't have many of the regularization term,
            this is to help it start at a decent place in a lower dimensional
            space, before going to the hard task of tweaking all at the same time.
        plateau_reduce_lr (list, optional): [patience, factor] If specified, if loss did not improve since `patience` epochs then multiply learning rate by `factor`.
        [None,None] means no reducing of lr on plateau.
        kwargs:
            Additional arguments to `get_seq2seq_model`.
    """
    logger.setLevel(log_level.upper())
    saved_args = locals()

    if torch.cuda.is_available():
        logging.info("Cuda device set to {}".format(cuda_device))
        torch.cuda.set_device(cuda_device)

    is_attn_field = get_is_attn_field(attender, loss_names, force_mu)

    train, dev, src, tgt, oneshot = get_train_dev(train_path,
                                                  dev_path,
                                                  max_len,
                                                  src_vocab,
                                                  tgt_vocab,
                                                  is_predict_eos=is_predict_eos,
                                                  is_add_attn=is_attn_field)
    logger.debug("is_add_attn: {}".format(is_attn_field))

    # When chosen to use attentive guidance, check whether the data is correct for the first
    # example in the data set. We can assume that the other examples are then also correct.
    if is_attn_field:
        is_dot_eos = train[0].__dict__['src'][-1] == "."
        if len(train) > 0:
            if 'attn' not in vars(train[0]):
                raise Exception("AttentionField not found in train data")
            tgt_len = len(vars(train[0])['tgt']) - 1  # -1 for SOS
            attn_len = len(vars(train[0])['attn']) - 1  # -1 for preprended ignore_index
            if attn_len != tgt_len and not (attn_len == tgt_len + 1 and is_dot_eos):
                raise Exception("Length of output sequence {} does not equal length of attention sequence in train data {}. First train example: {}".format(tgt_len, attn_len, train[0].__dict__))

        if dev is not None and len(dev) > 0:
            if 'attn' not in vars(dev[0]):
                raise Exception("AttentionField not found in dev data")
            tgt_len = len(vars(dev[0])['tgt']) - 1  # -1 for SOS
            attn_len = len(vars(dev[0])['attn']) - 1  # -1 for preprended ignore_index
            if attn_len != tgt_len and not (attn_len == tgt_len + 1 and is_dot_eos):
                raise Exception("Length of output sequence {} does not equal length of attention sequence in train data {}. First train example: {}".format(tgt_len, attn_len, train[0].__dict__))

    total_training_calls = math.ceil(epochs * len(train) / batch_size)
    rate2steps = Rate2Steps(total_training_calls)

    n_steps_prepare_pos = rate2steps(rate_prepare_pos)
    seq2seq = get_seq2seq_model(len(src.vocab), len(tgt.vocab), max_len, tgt.sos_id,
                                tgt.eos_id, total_training_calls,
                                content_method=content_method,
                                n_steps_prepare_pos=n_steps_prepare_pos,
                                attender=attender,
                                force_mu=force_mu,
                                **kwargs)

    n_parameters = sum([p.numel() for p in seq2seq.parameters()])
    saved_args["n_parameters"] = n_parameters

    seq2seq.reset_parameters()
    seq2seq.to(device)

    metrics = get_metrics(metric_names, src, tgt, is_predict_eos)

    max_p_interpolators = dict()
    max_p_interpolators.update(get_regularizers_location(total_training_calls,
                                                         n_steps_prepare_pos=n_steps_prepare_pos))
    losses, loss_weights = get_losses(loss_names, tgt, is_predict_eos,
                                      eos_weight=eos_weight,
                                      total_training_calls=total_training_calls,
                                      max_p_interpolators=max_p_interpolators,
                                      max_len=max_len)

    early_stopper = (EarlyStopping(patience=patience)
                     if patience is not None else None)

    if anneal_eos_weight != 0:
        n_steps_interpolate_eos_weight = rate2steps(anneal_eos_weight)

        loss_weight_updater = LossWeightUpdater(indices=[tgt.eos_id],
                                                initial_weights=[_initial_eos_weight],
                                                final_weights=[eos_weight],
                                                n_steps_interpolates=[0],
                                                # n_steps_interpolates=[n_steps_interpolate_eos_weight],
                                                modes=["linear"],
                                                start_steps=[n_steps_interpolate_eos_weight])  # DEV MODE
    else:
        loss_weight_updater = None

    trainer = SupervisedTrainer(loss=losses,
                                metrics=metrics,
                                loss_weights=loss_weights,
                                batch_size=batch_size,
                                eval_batch_size=eval_batch_size,
                                checkpoint_every=save_every,
                                print_every=print_every,
                                expt_dir=output_dir,
                                early_stopper=early_stopper,
                                loss_weight_updater=loss_weight_updater,
                                teacher_forcing=teacher_forcing,
                                initial_model=_initial_model,
                                log_level=log_level)

    optimizer_kwargs = {"max_grad_value": grad_clip_value,
                        "max_grad_norm": grad_clip_norm}

    if plateau_reduce_lr is not None:
        optimizer_kwargs["scheduler"] = ReduceLROnPlateau
        optimizer_kwargs["scheduler_kwargs"] = dict(patience=plateau_reduce_lr[0],
                                                    factor=plateau_reduce_lr[1])

    if (optim is None or optim == "adam") and is_amsgrad:
        optimizer_kwargs["amsgrad"] = True

    if pretrained_locator is None:
        optim_params = None
    else:
        _, loc_params = zip(*seq2seq.decoder.attender.named_params_locator())
        optim_params = [
            {"params": loc_params, "lr": 1e-4},  # 10x slower update
            {"params": p for p in seq2seq.parameters() if not
             any([p is loc_p for loc_p in loc_params])},
        ]

    seq2seq, history, other = trainer.train(seq2seq,
                                            train,
                                            num_epochs=epochs,
                                            dev_data=dev,
                                            optimizer=optim,
                                            optimizer_kwargs=optimizer_kwargs,
                                            learning_rate=lr,
                                            resume=resume,
                                            checkpoint_path=checkpoint_path,
                                            top_k=1,
                                            optim_params=optim_params)

    latest_model = get_latest_file(output_dir)
    if os.path.join(output_dir, _initial_model) == latest_model:
        shutil.copytree(latest_model, latest_model + " copy")

    if name_checkpoint is not None:
        _rename_latest_file(output_dir, name_checkpoint)
        final_model = os.path.join(output_dir, name_checkpoint)
    else:
        latest_model = get_latest_file(output_dir)
        final_model = os.path.join(output_dir, latest_model)

    # save the initial model to see initialization
    shutil.move(os.path.join(output_dir, _initial_model),
                os.path.join(final_model, _initial_model))

    _save_parameters(saved_args, final_model)

    return seq2seq, history, other
