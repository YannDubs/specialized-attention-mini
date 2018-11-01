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
from seq2seq.util.confuser import Confuser
from seq2seq.util.helpers import Rate2Steps, regularization_loss, get_latest_file
from seq2seq.attention.position import get_regularizers_positioner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
log_level = "warning"
logging.basicConfig(format=LOG_FORMAT, level=getattr(
    logging, log_level.upper()))
logger = logging.getLogger(__name__)


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
                      is_mlps=True,
                      embedding_size=128,
                      rnn_cell='gru',
                      hidden_size=128,
                      is_weight_norm=False,  # TO DO - medium: chose best and remove parameter + try layer normalization which should work better
                      dropout_input_encoder=0,  # TO DO - medium: give a single dictionary of kwargs defining `dropout_input_encoder`, `dropout_input_decoder`, `anneal_decoder_noise_input`. In constructor just intialize a list of each dropouts and then in `forward`, just call each of those.
                      dropout_input_decoder=0,
                      anneal_mid_dropout=0.1,  # TO DO - medium: give a single dictionary of kwargs defining `anneal_mid_dropout`, `anneal_mid_noise`.
                      anneal_mid_noise=0,
                      is_highway=True,
                      initial_highway=0.7,  # TO DO - medium: chose best and remove parameter
                      is_single_carry=True,
                      is_additive_highway=True,   # TO DO - medium: chose best and remove parameter
                      is_add_all_controller=True,
                      content_method='scalemult',  # see if scaledmult better
                      is_content_attn=True,
                      key_size=32,
                      value_size=-1,
                      is_contained_kv=False,
                      anneal_kq_dropout_output=0,
                      anneal_kq_noise_output=0.15,  # TO DO - medium: keep only one between `anneal_kq_dropout_output`, `anneal_kq_noise_output
                      is_position_attn=True,
                      n_steps_prepare_pos=None,
                      positioning_method="gaussian",
                      is_posrnn=True,
                      rate_init_help=0,
                      anneal_min_sigma=0.1,
                      is_bb_bias=True,
                      regularizations=["is_reg_clamp_mu", "is_l0_bb_weights", "is_reg_clamp_weights"],
                      lp_reg_weights=1,  # TO DO - medium: chose best and remove parameter
                      is_clamp_weights=True,  # TO DO - medium: chose best and remove parameter
                      rate_start_round=0,
                      anneal_temp_round=0.1,
                      rounder_weights="concrete",
                      rounder_mu="concrete",
                      mode_attn_mix="pos_conf",  # TO DO - medium: chose best and remove parameter
                      rate_attmix_wait=0,
                      default_pos_perc=0.5,  # TO DO - medium: chose best and remove parameter
                      rounder_perc="concrete",  # TO DO - medium: chose best and remove parameter
                      is_dev_mode=False,
                      is_viz_train=False):
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
        rnn_cell ({"gru", "lstm", optional): type of rnn.
        hidden_size (int, optional): hidden size for unidirectional encoder.
        is_weight_norm (bool, optional): whether to use weight normalization
            for the RNN. Weight normalization is similar to batch norm or layer
            normalization and has been shown to help when learning large models.
        dropout_input_encoder (float, optional): dropout after the embedding
            layer of the encoder.
        dropout_input_decoder (float, optional): dropout after the embedding
            layer of the decoder.
        anneal_mid_dropout (float, optional): annealed dropout between
            the decoder and encoder. `mid_dropout` will actually start at
            `initial_mid_dropout` and will geometrically decrease at each
            training calls, until it reaches `final_mid_dropout`. This
            parameter defines the percentage of training calls before the mdoel
            should reach `final_mid_dropout`.
        anneal_mid_noise (float, optional): annealed noise between
            the decoder and encoder. This parameter defines the percentage of
            training calls before the noise model should reach the final relative
            standard deviation.
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
        is_add_all_controller (bool, optional): whether to add all computed features
            to the decoder in order to have a central model that "knows everything".
        content_method ({"dot", "hard", "mlp"}, optional): content attention
            function to use.
        is_content_attn (bool, optional): whether to use content attention.
        key_size (int, optional): size of the generated key. -1 means same as hidden
            size. Can also give percentage of hidden size betwen 0 and 1.
        values_size (int, optional): size of the generated value. -1 means same
            as hidden size. Can also give percentage of hidden size betwen 0 and 1.
        is_contained_kv (bool, optional): whether to use different parts of the
            controller output as input for key and value generation.
        anneal_kq_dropout_output (float, optional): annealed dropout to
            the output of the key and query generator. This parameter
            defines the percentage of training calls before the model should reach
            the final dropout.
        anneal_kq_noise_output (float, optional): annealed noise to
            the output of the key and query generator. This parameter
            defines the percentage of training calls before the noise model should
            reach the final relative standard deviation.
        is_position_attn (bool, optional): whether to use positional attention.
        n_steps_prepare_pos (int, optional): number steps during which
            to consider the positioning as in a preparation mode. During
            preparation mode, the model have less parameters to tweak, it will
            focus on what I thought were the most crucial bits. For example it
            will have a fix sigma and won't have many of the regularization term,
            this is to help it start at a decent place in a lower dimensional
            space, before going to the hard task of tweaking all at the same time.
        positioning_method ({"gaussian",
            "laplace"}, optional): name of the positional distribution.
            `laplace` is more human plausible but `gaussian` works best.
        is_posrnn (bool, optional): whether to use a rnn for the positional
            attention generator.
        rate_init_help (float, optional): percentage of total steps for which to
            a initializer helper for the position attention. Currently the helper
            consists of alternating between values of 0.5 and -0.5 for the
            "rel_counter_decoder" weights.
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
        regularizations (list of str, optional): list of regularizations to use.
            Possible regularizations Are explained below :
            - "is_reg_const_weights" : whether to use a lp regularization
                on the constant position mu building block. This can be usefull in
                otrder to push the network to use non constant building blocks that are
                more extrapolable (i.e with constants, the network has to make varying
                weights which is not interpretable. If the blocks ae varying then
                the "hard" extrapolable output would already be done for the network).
            - "is_reg_old_weights": whether to use a lp norm regularisation
                on the building blocks that depend on previous positioning attention.
                This can be useful as these building blocks cannot be used correctly
                before positioning attention actually converged.
            - "is_reg_clamp_mu": whether to regularise with lp norm the
                clamping of mu. I.e push the network to not overshoot and really
                generate the desired mu rather than the clamped one. This can be
                useful as if the mu completely overshoots it will be hard for it to
                come back to normal values if it needs to. It also makes sense to
                output what you want rather than relying on postpropressing.
            - "is_reg_round_weighs": whether to regularise with lp norm
                the building block weights in order to push them towards integers.
                This is the soft version of `rounder_weights`.
            - "is_reg_variance_weights": whether to use lp norm
                regularisation to force the building blocks to have low variance
                across time steps. This can be useful as it forces the model to use
                simpler weight patterns that are more extrapolable. For example it
                would prefer giving a weight of `1` to `block_j/n`than using a weight
                of `j` to `block_1/n`.
            - "is_l0_bb_weights": whether to use l0 regularisation on
                the building block weights. This is achieved by reparametrizing the
                l0 loss as done in “Learning Sparse Neural Network through L_0
                Regularisation”.
            - "is_reg_pos_perc": whether to use lp norm regularisation
                in order to push the network to use positional attention when it can.
                This is desirable as positional attention is tailored for location
                attention and is thus more interpretable and extrapolable. This is
                only needed if content attention is able to find some positional
                pattern, which shouldn’t be the case if it confused correctly.
        lp_reg_weights (bool, optional): the p in the lp norm to use for all the
            regularisation above. p can be in [0,”inf”]. If `p=0` will use some
            approximation to the l0 norm.
        is_clamp_weights (bool, optional): whether to clamp the building block
            weights on some meaningful intervals.
        rate_start_rounding (float, optional): percentage of training steps to
            wait before starting the rounding of all variables to round.
        anneal_temp_round (float, optional): percentage of training steps for
            which to anneal the temperature in the rounding of all variables to round.
        rounder_weights ({“concrete”, “stochastic”, None}, optional): the method
            for approximative differential rounding to use for the required building
            block weights. If `concrete` it will use the concrete distribution
            similarly to “The Concrete Distribution: A Continuous Relaxation of
            Discrete Random Variables”. If `stochastic` it will round to the
            ceiling with a probability equal to the decimal of x and then use
            straight through estimators in the backward pass. If `None` will not
            use any rounding. Rounding is desirable to make the output more
            interpretable and extrapolable (as the building blocks were designed
            such that integer wights could be used to solve most positonal patterns).
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

    # Encoder
    kq_annealed_dropout_output_kwargs = dict(
        n_steps_interpolate=rate2steps(anneal_kq_dropout_output))
    kq_annealed_noise_output_kwargs = dict(
        n_steps_interpolate=rate2steps(anneal_kq_noise_output))

    key_kwargs = dict(output_size=key_size,
                      is_contained_kv=is_contained_kv,
                      is_mlps=is_mlps,
                      annealed_dropout_output_kwargs=kq_annealed_dropout_output_kwargs,
                      annealed_noise_output_kwargs=kq_annealed_noise_output_kwargs)

    highway_kwargs = dict(initial_highway=initial_highway,
                          is_single_carry=is_single_carry,
                          is_additive_highway=is_additive_highway,
                          is_mlps=is_mlps)

    value_kwargs = dict(output_size=value_size,
                        is_contained_kv=is_contained_kv,
                        is_highway=is_highway,
                        highway_kwargs=highway_kwargs)

    encoder = EncoderRNN(len(src.vocab),
                         max_len,
                         hidden_size,
                         embedding_size,
                         input_dropout_p=dropout_input_encoder,
                         rnn_cell=rnn_cell,
                         is_weight_norm=is_weight_norm,
                         key_kwargs=key_kwargs,
                         value_kwargs=value_kwargs)

    # Decoder
    query_kwargs = key_kwargs  # use the same parameters

    n_steps_start_round = rate2steps(rate_start_round)
    rounders_kwars = {"concrete": {"n_steps_interpolate": rate2steps(anneal_temp_round),
                                   "start_step": n_steps_start_round},
                      "stochastic": {"start_step": n_steps_start_round},
                      None: {}}

    rounder_weights_kwargs = dict(name=rounder_weights)
    rounder_weights_kwargs.update(rounders_kwars[rounder_weights])

    rounder_mu_kwargs = dict(name=rounder_mu)
    rounder_mu_kwargs.update(rounders_kwars[rounder_mu])

    rnn_kwargs = dict(is_weight_norm=is_weight_norm)
    position_kwargs = dict(n_steps_prepare_pos=n_steps_prepare_pos,
                           n_steps_init_help=rate2steps(rate_init_help),
                           is_recurrent=is_posrnn,
                           positioning_method=positioning_method,
                           n_steps_interpolate_min_sigma=rate2steps(
                               anneal_min_sigma),
                           rnn_cell=rnn_cell,
                           is_mlps=is_mlps,
                           rnn_kwargs=rnn_kwargs,
                           is_bb_bias=is_bb_bias,
                           regularizations=regularizations,
                           is_clamp_weights=is_clamp_weights,
                           rounder_weights_kwargs=rounder_weights_kwargs,
                           rounder_mu_kwargs=rounder_mu_kwargs)

    content_kwargs = dict(method=content_method)

    rounder_perc_kwargs = dict(name=rounder_perc)
    rounder_perc_kwargs.update(rounders_kwars[rounder_perc])

    n_steps_wait = rate2steps(rate_attmix_wait)
    attmix_kwargs = dict(is_mlps=is_mlps,
                         mode=mode_attn_mix,
                         n_steps_wait=n_steps_wait,
                         rounder_perc_kwargs=rounder_perc_kwargs,
                         is_reg_pos_perc="is_reg_pos_perc" in regularizations,
                         default_pos_perc=default_pos_perc)

    decoder = DecoderRNN(len(tgt.vocab),
                         max_len,
                         hidden_size,
                         embedding_size,
                         tgt.sos_id,
                         tgt.eos_id,
                         input_dropout_p=dropout_input_decoder,
                         rnn_cell=rnn_cell,
                         is_weight_norm=is_weight_norm,
                         value_size=encoder.value_size,
                         is_content_attn=is_content_attn,
                         is_position_attn=is_position_attn,
                         content_kwargs=content_kwargs,
                         position_kwargs=position_kwargs,
                         query_kwargs=query_kwargs,
                         attmix_kwargs=attmix_kwargs,
                         is_add_all_controller=is_add_all_controller)

    mid_dropout_kwargs = dict(
        n_steps_interpolate=rate2steps(anneal_mid_dropout))
    mid_noise_kwargs = dict(n_steps_interpolate=rate2steps(anneal_mid_noise))

    seq2seq = Seq2seq(encoder, decoder,
                      mid_dropout_kwargs=mid_dropout_kwargs,
                      mid_noise_kwargs=mid_noise_kwargs)

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
          content_method='scalemult',
          is_amsgrad=True,  # TO DO - medium : chose best valeu and delete param
          rate_prepare_pos=0.05,
          is_confuse_key=False,
          key_generator_criterion="l05",  # TO DO - medium : chose best valeu and delete param
          is_confuse_query=False,
          query_generator_criterion="l05",  # TO DO - medium : chose best valeu and delete param
          n_steps_discriminate_only=15,
          n_steps_interpolate_confuser=0.05,
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
        is_confuse_key (bool, optional): whether to remove the ability of the
            key to know what encoding step it is at. By doing so the network is
            forced to used the positional attention when counting is crucial.
        key_generator_criterion ({"l05", "l1"}, optional): what type of loss
            to use for the confusers key generator.
        is_confuse_query (bool, optional): whether to remove the ability of the
            query to know what decoding step it is at. By doing so the network is
            forced to used the positional attention when counting is crucial.
        query_generator_criterion ({"l05", "l1"}, optional): what type of loss
            to use for the confusers key generator.
        n_steps_discriminate_only (int, optional): Number of steps at the begining
           where you only train the discriminator.
        n_steps_interpolate_confuser (int, optional): number of interpolating steps before
            reaching the `final_factor` and `final_max_scale`.
        plateau_reduce_lr (list, optional): [patience, factor] If specified, if loss did not improve since `patience` epochs then multiply learning rate by `factor`.
        [None,None] means no reducing of lr on plateau.
        kwargs:
            Additional arguments to `get_seq2seq_model`.
    """
    saved_args = locals()
    logger.setLevel(log_level.upper())

    if torch.cuda.is_available():
        print("Cuda device set to %i" % cuda_device)
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
    max_p_interpolators.update(get_regularizers_positioner(total_training_calls,
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

    confusers = dict()
    if is_confuse_key:
        # don't confuse the whole model, only the key generator
        generator = seq2seq.encoder.key_generator
        if key_generator_criterion == "l05":
            generator_criterion = _l05loss
        elif key_generator_criterion == "l1":
            generator_criterion = nn.L1Loss(reduction="none")
        confusers["key_confuser"] = Confuser(nn.L1Loss(reduction="none"),
                                             seq2seq.encoder.key_size + 1,  # will add n
                                             generator_criterion=generator_criterion,
                                             target_size=1,
                                             n_steps_discriminate_only=rate2steps(n_steps_discriminate_only),
                                             generator=generator,
                                             n_steps_interpolate=rate2steps(n_steps_interpolate_confuser))

    if is_confuse_query:
        # don't confuse the whole model, only the query generator
        generator = seq2seq.decoder.query_generator
        if query_generator_criterion == "l05":
            generator_criterion = _l05loss
        elif query_generator_criterion == "l1":
            generator_criterion = nn.L1Loss(reduction="none")
        confusers["query_confuser"] = Confuser(nn.L1Loss(reduction="none"),
                                               seq2seq.decoder.query_size + 1,  # will add n
                                               generator_criterion=generator_criterion,
                                               target_size=1,
                                               n_steps_discriminate_only=rate2steps(n_steps_discriminate_only),
                                               generator=generator,
                                               n_steps_interpolate=rate2steps(n_steps_interpolate_confuser))

    for _, confuser in confusers.items():
        confuser.to(device)

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
                                            confusers=confusers)
    # DEV MODE
    other["confusers"] = confusers

    for _, confuser in other["confusers"].items():
        confuser.to(torch.device("cpu"))

    if oneshot is not None:
        (seq2seq,
         history_oneshot,
         other) = trainer.train(seq2seq,
                                oneshot,
                                num_epochs=5,
                                dev_data=dev,
                                optimizer=optim,
                                optimizer_kwargs=optimizer_kwargs,
                                teacher_forcing_ratio=0,
                                learning_rate=lr,
                                is_oneshot=True,
                                checkpoint_path=get_latest_file(output_dir),
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
