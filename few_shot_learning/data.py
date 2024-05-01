from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp


######################################################################
## data generation
######################################################################

import numpy as np

def create_reg_data(rng, i_size, c_size, size_distract, input_range, w_scale):
    """
    Create a linear regression data set: X*w where x ~ U(-1, 1), w ~ N(0,1).
    x, y are concatenated
    """

    w = rng.normal(size=[i_size]) * w_scale

    x = rng.uniform(
        low=-input_range / 2, high=input_range / 2, size=[c_size, i_size]
    )
    x_querry = rng.uniform(
        low=-input_range / 2, high=input_range / 2, size=[1, i_size]
    )

    y_data = np.squeeze(np.matmul(x, w))
    choice = rng.choice(c_size, size=[size_distract], replace=False)
    y_data[choice] = rng.normal(size=[size_distract])

    y_target = np.matmul(x_querry, w)
    y_target = y_target[..., None]

    seq = np.concatenate([x, y_data[..., None]], -1)
    target = np.concatenate([x_querry, y_target], -1)
    x_querry_init = -1 * np.matmul(x_querry, np.ones_like(x_querry).T * 0.0)
    zero = np.concatenate([x_querry, x_querry_init], -1)
    seq = np.concatenate([seq, zero], 0)
    return np.squeeze(seq), np.squeeze(target), w


def get_reg_data(
    no_tasks, feature_size, no_examples, size_distract=0, input_range=1, w_scale=1
):
    """
    Create a linear regression data set: X*w where x ~ U(-1, 1), w ~ N(0,1).
    x, y are concatenated
    Note that for HW6, one only needs a single dataset like eval_data in constructed_token_setup
    """
    rng = np.random.default_rng()
    seqs, targets, ws = [], [], []
    for _ in range(no_tasks):
        seq, target, w = create_reg_data(
            rng, feature_size, no_examples, size_distract, input_range, w_scale
        )
        seqs.append(seq)
        targets.append(target)
        ws.append(w)

    return np.array(seqs), np.array(targets), np.array(ws)


######################################################################
## weights for linear self attention
######################################################################


def create_weights(
    feature_size,
    output_size,
    c_size,
    lr,
    w_init=None,
    second_zero=False,
    lin_diag=False,
    gd_deq=False,
    num_layers=1,
):
    """
    Create linear regression gradient descent weights for self-attention
    layer.
    param feature_size: int, size of the input
    param output_size: int, size of the output
    param c_size: int, size of the context (samples N in HW6)
    param lr: float, learning rate
    param w_init: list of np.array, initial weights for the linear layer
    param second_zero: bool, if True, the second half of the output is zero
    param lin_diag: bool, if True, the diagonal of the value matrix is linear
    param gd_deq: bool, if True, the weights are for the gradient descent
    param num_layers: int, number of layers in the transformer
    """
    if w_init is None:
        w_init = np.random.normal(size=[1, 1, feature_size]) * 0

    one = np.ones([feature_size + output_size])
    one_in_size = np.ones([feature_size])
    zero_out_size = np.zeros([output_size])
    one_out_size = np.ones([output_size])

    # Value matrix
    value_upper = np.zeros([feature_size, feature_size + output_size])
    value_lower_left = w_init[0]
    if lin_diag:
        value_lower_right = np.diag(one_out_size) * -2
    else:
        value_lower_right = np.diag(one_out_size) * 1

    if second_zero:
        value_lower_right = np.diag(zero_out_size)

    value_lower_part = np.concatenate([value_lower_left, value_lower_right], axis=1)
    value_matrix = np.concatenate([value_upper, value_lower_part], axis=0).T
    if lin_diag:
        value_matrix += np.diag(one)

    # Query and Key matrix
    query_upper_part = np.zeros([output_size, feature_size + output_size])
    query_lower_left = np.diag(one_in_size)
    query_lower_right = np.zeros([feature_size, output_size])
    query_lower_part = np.concatenate([query_lower_left, query_lower_right], axis=1)
    query_matrix = np.concatenate([query_lower_part, query_upper_part], axis=0)
    key_matrix = query_matrix

    # Projection matrix
    projection_upper_part = np.zeros([feature_size, feature_size + output_size])
    projection_lower_left = np.zeros([output_size, feature_size])

    projection_lower_right = np.diag(one_out_size) * ((1 / c_size) * lr)

    if lin_diag:
        shifted_lr = np.diag(one_out_size) * (1 / c_size) * (1 / c_size) * lr
        projection_lower_right += shifted_lr

    projection_lower_part = np.concatenate(
        [projection_lower_left, projection_lower_right], axis=1
    )
    projection_matrix = np.concatenate(
        [projection_upper_part, projection_lower_part], axis=0
    )
    if lin_diag:
        projection_matrix -= np.diag(one) * (1 / c_size) * (1 / c_size) * lr

    params_new = {}
    for layer in range(num_layers):
        if num_layers == 1 or gd_deq:
            tra_name = "Transformer_gd/multi_head_attention/"
        else:
            tra_name = "Transformer_gd/~trans_block/layer_" + str(layer) + "/"
        params_new[tra_name + "query"] = {"w": np.array(query_matrix)}
        params_new[tra_name + "value"] = {"w": np.array(value_matrix)}
        params_new[tra_name + "key"] = {"w": np.array(key_matrix)}
        params_new[tra_name + "linear"] = {"w": np.array(projection_matrix)}

    return params_new

