"""
Main package entrypoint.

TO - DO:
- There are way too many arguments that I have kept to test, you should not
    keep all of them when refactoring for self-attention. Focus On the important
    ones.
- there are many different. Should not give all in different parameters but simply
    a dictionary. And in constructor just initialize a list of each dropouts / noising
    classes and in `forward` just call each
- should modulrize more. Ex plit : gettseq2seq. Into get encoder / decoder / attn
"""

import os
import logging
import warnings
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
from seq2seq.attention import Attender, LocationOnlyAttender, ContentOnlyAttender
from seq2seq.attention.location import get_regularizers_location

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
log_level = "warning"
logging.basicConfig(format=LOG_FORMAT, level=getattr(
    logging, log_level.upper()))
logger = logging.getLogger(__name__)


def get_attender(name, attender_kwargs):
    if name == "attender":
        return dict(attender=Attender, attender_kwargs=attender_kwargs)
    elif name == "location":
        return dict(attender=LocationOnlyAttender,
                    attender_kwargs=attender_kwargs["location_kwargs"])
    elif name == "content":
        return dict(attender=ContentOnlyAttender,
                    attender_kwargs=attender_kwargs["content_kwargs"])
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


def get_seq2seq_model(src,
                      tgt,
                      max_len,
                      total_training_calls,
                      is_mlps=False,
                      embedding_size=64,
                      hidden_size=128,
                      anneal_mid_dropout=0.3,
                      mid_noise_sigma=0,
                      is_highway=True,
                      initial_gate=0.7,  # TO DO - medium: chose best and remove parameter
                      is_single_gate=True,
                      is_additive_highway=True,   # TO DO - medium: chose best and remove parameter
                      content_method='scaledot',  # see if scaledmult better
                      value_size=-1,
                      value_noise_sigma=0,
                      n_steps_prepare_pos=None,
                      positioning_method="gaussian",
                      anneal_min_sigma=0.1,
                      rate_start_round=0.05,
                      anneal_temp_round=0.1,
                      rounder_mu="concrete",
                      mode_attn_mix="loc_conf",  # TO DO - medium: chose best and remove parameter
                      rate_attmix_wait=0,
                      default_pos_perc=0.5,  # TO DO - medium: chose best and remove parameter
                      rounder_perc="concrete",  # TO DO - medium: chose best and remove parameter
                      is_dev_mode=False,
                      is_viz_train=False,
                      attender="attender",
                      is_leaky_noisy_clamp=False,  # DEV MODE
                      is_l0_bb_weights=True,  # DEV MODE
                      is_reg_clamp_mu=True,  # DEV MODE
                      gating="custom",  # DEV MODE
                      rounder_weights=None  # DEV MODE
                      ):
    """Return a initialized extrapolator model.

    Args:
        src (SourceField): source field.
        tgt (TargetField): target field.
        max_len (int): maximum possible length of any source sentence.
        total_training_calls (int): number of maximum training calls.
        is_mlps (bool, optional): whether to use MLPs for the generators instead
            of a linear layer.
        embedding_size (int, optional): size of embedding for the decoder and
            encoder.
        hidden_size (int, optional): hidden size for unidirectional encoder.
        anneal_mid_dropout (float, optional): annealed dropout between
            the decoder and encoder. `mid_dropout` will actually start at
            `initial_mid_dropout` and will geometrically decrease at each
            training calls, until it reaches `final_mid_dropout`. This
            parameter defines the percentage of training calls before the mdoel
            should reach `final_mid_dropout`.
        mid_noise_sigma (float, optional) relative noise to add between the decoder
            and encoder. This can be seen as a rough approximation to building a
            variational latent representation as it forces the prediction of a
            distribution rather than points.
        is_highway (bool, optional): whether to use a highway betwen the embedding
            and the value of the encoder.
        initial_highway (float, optional): initial highway carry rate. This can be
            useful to make the network learn the attention even before the
            decoder converged.
        is_single_carry (bool, optional): whetehr to use a one dimension carry weight
            instead of n dimensional. If a n dimension then the network can learn
            to carry some dimensions but not others. The downside is that
            the number of parameters would be larger.
        is_additive_highway (bool, optional): whether to use a residual connection
            with a carry weight got th residue. I.e if `True` the carry weight will
            only be applied to the residue and will not scale the new value with
            `1-carry`.
        content_method ({"dot", "hard", "mlp"}, optional): content attention
            function to use.
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
        anneal_min_sigma (float, optional): if not 0 , it will force
            the network to keep a higher sigma while it's learning. min_sigma will
            actually start at `initial_sigma` and will linearly decrease at each
            training calls, until it reaches the given `min_sigma`. This parameter
            defines the percentage of training calls before the mdoel should reach
            the final `min_sigma`.
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
            will use `default_pos_perc`.
        default_pos_perc (float, optional): constant positional percentage to
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
    """
    assert max_len > 1, "Max len has to be greater than 1"

    # interpolating rates to interpolating steps
    rate2steps = Rate2Steps(total_training_calls)

    Generator = MLP if is_mlps else nn.Linear

    # Encoder
    min_hidden = 16
    highway_kwargs = dict(initial_gate=initial_gate,
                          is_single_gate=is_single_gate,
                          is_additive_highway=is_additive_highway,
                          is_mlps=is_mlps,
                          min_hidden=min_hidden)

    value_kwargs = dict(output_size=value_size,
                        is_highway=is_highway,
                        highway_kwargs=highway_kwargs,
                        sigma_noise=value_noise_sigma)

    encoder = EncoderRNN(len(src.vocab),
                         max_len,
                         hidden_size,
                         embedding_size,
                         value_kwargs=value_kwargs)

    # Decoder
    n_steps_start_round = rate2steps(rate_start_round)
    rounders_kwars = {"concrete": {"n_steps_interpolate": rate2steps(anneal_temp_round),
                                   "start_step": n_steps_start_round},
                      "stochastic": {"start_step": n_steps_start_round},
                      None: {}}

    rounder_mu_kwargs = dict(name=rounder_mu)
    rounder_mu_kwargs.update(rounders_kwars[rounder_mu])
    rounder_weights_kwargs = dict(name=rounder_weights)
    rounder_weights_kwargs.update(rounders_kwars[rounder_weights])

    mu_kwargs = dict(rounder_mu_kwargs=rounder_mu_kwargs,
                     is_leaky_noisy_clamp=is_leaky_noisy_clamp,
                     is_l0_bb_weights=is_l0_bb_weights,
                     is_reg_clamp_mu=is_reg_clamp_mu,
                     rounder_weights_kwargs=rounder_weights_kwargs)

    location_kwargs = dict(n_steps_prepare_pos=n_steps_prepare_pos,
                           pdf=positioning_method,
                           Generator=Generator,
                           mu_kwargs=mu_kwargs,
                           gating=gating)

    content_kwargs = dict(scorer=content_method)

    rounder_perc_kwargs = dict(name=rounder_perc)
    rounder_perc_kwargs.update(rounders_kwars[rounder_perc])

    n_steps_wait = rate2steps(rate_attmix_wait)
    mixer_kwargs = dict(is_mlps=is_mlps,
                        mode=mode_attn_mix,
                        n_steps_wait=n_steps_wait,
                        rounder_perc_kwargs=rounder_perc_kwargs,
                        default_pos_perc=default_pos_perc)

    attender_kwargs = dict(content_kwargs=content_kwargs,
                           location_kwargs=location_kwargs,
                           mixer_kwargs=mixer_kwargs)

    decoder = DecoderRNN(len(tgt.vocab),
                         max_len,
                         hidden_size,
                         embedding_size,
                         tgt.sos_id,
                         tgt.eos_id,
                         value_size=encoder.value_size,
                         **get_attender(attender, attender_kwargs))

    mid_dropout_kwargs = dict(n_steps_interpolate=rate2steps(anneal_mid_dropout))

    seq2seq = Seq2seq(encoder, decoder,
                      mid_dropout_kwargs=mid_dropout_kwargs,
                      mid_noise_sigma=mid_noise_sigma)

    seq2seq.set_dev_mode(value=is_dev_mode)
    seq2seq.set_viz_train(value=is_viz_train)

    return seq2seq


def train(train_path,
          dev_path,
          oneshot_path=None,
          metric_names=["word accuracy", "sequence accuracy",
                        "final target accuracy"],
          loss_names=["nll"],
          max_len=50,
          epochs=100,
          output_dir="models/",
          src_vocab=50000,
          tgt_vocab=50000,
          is_predict_eos=True,
          anneal_teacher_forcing=0,
          initial_teacher_forcing=0.2,
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
          is_attnloss=False,
          eos_weight=1,
          anneal_eos_weight=0,  # TO DO : hyperparmeter optimize
          _initial_eos_weight=0.05,
          content_method='scaledot',
          is_amsgrad=True,  # TO DO - medium : chose best valeu and delete param
          rate_prepare_pos=0.05,
          plateau_reduce_lr=[4, 0.5],
          _initial_model="initial_model",
          **kwargs):
    """Trains the model given all parameters.

    Args:
        train_path (str): path to the training data.
        dev_path (str): path to the validation data.
        oneshot_path (str, optional): path to the data containing the new examples
            that should be learned in a few shot learning. If given, the model
            will be transfered on this data after having converged on the training
            data.
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
        anneal_teacher_forcing (float, optional): annealed teacher forcing,
            the teacher forcing will start at `initial_teacher_forcing` and will
            linearly decrease at each training calls, until it reaches 0. This
            parameter defines the percentage  of training calls before reaching
            a teacher_forcing_ratio of 0.
        initial_teacher_forcing (float, optional): initial teacher forcing ratio.
            If `anneal_teacher_forcing==0` will be the constant teacher forcing ratio.
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
        is_attnloss (str, optional): Whether to add attention loss, to force the
            netwrok to learn the given attention, as seen in "Learning
            compositionally through attentive guidance".
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
    saved_args = locals()
    logger.setLevel(log_level.upper())

    if torch.cuda.is_available():
        print("Cuda device set to {}".format(cuda_device))
        torch.cuda.set_device(cuda_device)

    train, dev, src, tgt, oneshot = get_train_dev(train_path,
                                                  dev_path,
                                                  max_len,
                                                  src_vocab,
                                                  tgt_vocab,
                                                  is_predict_eos=is_predict_eos,
                                                  content_method=content_method,
                                                  oneshot_path=oneshot_path)

    total_training_calls = math.ceil(epochs * len(train) / batch_size)
    rate2steps = Rate2Steps(total_training_calls)

    n_steps_prepare_pos = rate2steps(rate_prepare_pos)
    seq2seq = get_seq2seq_model(src, tgt, max_len, total_training_calls,
                                content_method=content_method,
                                n_steps_prepare_pos=n_steps_prepare_pos,
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
                                      max_p_interpolators=max_p_interpolators)

    early_stopper = (EarlyStopping(patience=patience)
                     if patience is not None else None)

    if anneal_eos_weight != 0:
        n_steps_interpolate_eos_weight = rate2steps(anneal_eos_weight)

        loss_weight_updater = LossWeightUpdater(indices=[tgt.eos_id],
                                                initial_weights=[_initial_eos_weight],
                                                final_weights=[eos_weight],
                                                n_steps_interpolates=[n_steps_interpolate_eos_weight],
                                                modes=["geometric"])
    else:
        loss_weight_updater = None

    final_teacher_forcing = 0 if anneal_teacher_forcing != 0 else initial_teacher_forcing
    teacher_forcing_kwargs = dict(initial_value=initial_teacher_forcing,
                                  final_value=final_teacher_forcing,
                                  n_steps_interpolate=rate2steps(anneal_teacher_forcing),
                                  mode="linear")

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
                                teacher_forcing_kwargs=teacher_forcing_kwargs,
                                initial_model=_initial_model)

    optimizer_kwargs = {"max_grad_value": grad_clip_value,
                        "max_grad_norm": grad_clip_norm}

    if plateau_reduce_lr is not None:
        optimizer_kwargs["scheduler"] = ReduceLROnPlateau
        optimizer_kwargs["scheduler_kwargs"] = dict(patience=plateau_reduce_lr[0],
                                                    factor=plateau_reduce_lr[1])

    if (optim is None or optim == "adam") and is_amsgrad:
        optimizer_kwargs["amsgrad"] = True

    seq2seq, history, other = trainer.train(seq2seq,
                                            train,
                                            num_epochs=epochs,
                                            dev_data=dev,
                                            optimizer=optim,
                                            optimizer_kwargs=optimizer_kwargs,
                                            learning_rate=lr,
                                            resume=resume,
                                            checkpoint_path=checkpoint_path,
                                            top_k=1)

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


def _l05loss(pred, target):
    return regularization_loss(pred - target, is_no_mean=True, p=0.5)
