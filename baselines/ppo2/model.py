import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
from baselines.common.input import observation_placeholder
from flow.envs.multi_merge import NoneBox

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None


def reduce_weighted_mean(x, mask=None, axis=None):
    if mask is None:
        return tf.reduce_mean(x)
    return tf.reduce_sum(x * mask, axis) / tf.reduce_sum(mask, axis)


def maybe_squeeze(tensor, axis=-1):
    if tensor.shape[axis] == 1:
        return tf.squeeze(tensor, axis)
    return tensor


class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, microbatch_size=None):
        self.sess = sess = get_session()

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            obs_ph = observation_placeholder(ob_space)
            # if isinstance(ob_space, NoneBox):
            #     obs_ph = observation_placeholder(ob_space)
            # else:
            #     obs_ph = None
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess, obs_ph)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess, obs_ph)
            else:
                train_model = policy(microbatch_size, nsteps, sess, obs_ph)

        # masks for losses
        obs_tensor = train_model.X
        if len(obs_tensor.shape) == 3:
            pad_vehs = maybe_squeeze(train_model.pad_vehicles)
            pad_timesteps = tf.reduce_all(pad_vehs, axis=-1)

            tf.summary.histogram('observation', obs_tensor, family='inputs')
            pad_obs = tf.boolean_mask(obs_tensor, pad_vehs)
            tf.summary.histogram('pad_observations', pad_obs, family='inputs')

            # self.zero = tf.to_float(zero_obs)
            self.nonpad_vehs = maybe_squeeze(train_model.nonpad_vehicles)
            self.nonpad_vehs_float = maybe_squeeze(train_model.nonpad_vehicles_float)
            self.nonpad_timesteps = tf.to_float(tf.logical_not(pad_timesteps))

            nonpad_obs = tf.boolean_mask(obs_tensor, self.nonpad_vehs)
            nonpad_obs_slices = tf.unstack(nonpad_obs, axis=-1)
            for t, tensor in enumerate(nonpad_obs_slices):
                tf.summary.histogram('nonpad_obs_{}'.format(t), tensor, family='inputs')
        else:
            self.nonpad_timesteps = None
            self.nonpad_vehs_float = None

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        tf.summary.histogram('action', A, family='inputs')
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        tf.summary.histogram('advantages', ADV, family='values')
        tf.summary.scalar('mean_advantage', tf.reduce_mean(ADV), family='values')
        self.R = R = tf.placeholder(tf.float32, [None])
        tf.summary.histogram('returns', R, family='values')
        tf.summary.scalar('mean_returns', tf.reduce_mean(R), family='values')
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        # vf clip range
        self.VF_CLIPRANGE = VF_CLIPRANGE = tf.placeholder(tf.float32, [])

        try:
            neglogpac = train_model.pd.neglogp(A, self.nonpad_vehs_float)
        except TypeError:
            neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        try:
            entropy = train_model.pd.entropy(self.nonpad_vehs_float)
        except TypeError:
            entropy = train_model.pd.entropy()
        entropy = reduce_weighted_mean(entropy, self.nonpad_timesteps)

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        tf.summary.histogram('predicted_value', vpred, family='value_fn')
        tf.summary.histogram('true_value', R, family='value_fn')
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - VF_CLIPRANGE, VF_CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        tf.summary.histogram('unclipped_value_loss', vf_losses1, family='value_fn')
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)
        tf.summary.histogram('clipped_value_loss', vf_losses2, family='value_fn')

        vf_loss = .5 * reduce_weighted_mean(tf.maximum(vf_losses1, vf_losses2),
                                            self.nonpad_timesteps)

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        tf.summary.histogram('ratio', ratio, family='losses')
        tf.summary.scalar('mean_ratio', tf.reduce_mean(ratio), family='losses')

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = reduce_weighted_mean(tf.maximum(pg_losses, pg_losses2),
                                       self.nonpad_timesteps)
        approxkl = .5 * reduce_weighted_mean(tf.square(neglogpac - OLDNEGLOGPAC),
                                             self.nonpad_timesteps)
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        for l, name in zip(self.stats_list, self.loss_names):
            tf.summary.scalar(name, l, family='losses')

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None,
              vf_cliprange=None, normalize_advantages=True):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        if normalize_advantages:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        if vf_cliprange is None:
            vf_cliprange = cliprange

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.VF_CLIPRANGE: vf_cliprange
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

    def summary_op(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs,
                   vf_cliprange=None, summary_op=None, normalize_advantages=True):
        if summary_op is None:
            return
        advs = returns - values

        # Normalize the advantages
        if normalize_advantages:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        if vf_cliprange is None:
            vf_cliprange = cliprange

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.VF_CLIPRANGE: vf_cliprange
        }

        return self.sess.run(summary_op, td_map)
