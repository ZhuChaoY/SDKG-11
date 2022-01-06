import re
import tensorflow as tf


def AdamWeightDecayOptimizer(loss, l_r, train_steps, warmup_steps):
    """Creates an optimizer training op."""
    
    global_step = tf.train.get_or_create_global_step()
    l_r = tf.constant(l_r)
    l_r = tf.train.polynomial_decay(l_r, global_step, train_steps,
          end_learning_rate = 0.0, power = 1.0, cycle = False)
    is_warmup = tf.cast(global_step < warmup_steps, tf.float32)
    l_r = (1.0 - is_warmup) * l_r + \
           is_warmup * l_r * tf.cast(global_step, tf.float32) / \
           tf.cast(warmup_steps, tf.float32)
    optimizer = AdamWeightDecay(l_r)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm = 1.0)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)
    train_op = tf.group(train_op, [global_step.assign(global_step + 1)])
    return train_op


class AdamWeightDecay(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self, l_r):
        """Constructs a AdamWeightDecayOptimizer."""

        super().__init__(False, 'AdamWeightDecayOptimizer')
        self.l_r = l_r
        self.w_decay_rate = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-6


    def apply_gradients(self, grads_and_vars, global_step = None):
        """See base class."""
    
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue
            param_name = self._get_variable_name(param.name)
            m = tf.get_variable(param_name + '/adam_m', param.shape.as_list(),
                tf.float32, tf.zeros_initializer(), trainable = False)
            v = tf.get_variable(param_name + '/adam_v', param.shape.as_list(),
                tf.float32, tf.zeros_initializer(), trainable = False)
            next_m = tf.multiply(self.beta_1, m) + \
                     tf.multiply(1.0 - self.beta_1, grad)
            next_v = tf.multiply(self.beta_2, v) + \
                     tf.multiply(1.0 - self.beta_2, tf.square(grad))
            update = next_m / (tf.sqrt(next_v) + self.epsilon)
            if self._do_use_weight_decay(param_name):
                update += self.w_decay_rate * param
            update_with_lr = self.l_r * update
            next_param = param - update_with_lr
            assignments.extend([param.assign(next_param), m.assign(next_m),
                                v.assign(next_v)])
        return tf.group(*assignments, name = None)


    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""

        for r in ['LayerNorm', 'layer_norm', 'bias']:
            if re.search(r, param_name) is not None:
                return False
        return True


    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

