from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp
import pandas as pd


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


def create_nonlinear_data(rng, c_size, i_size = 10):
    """
    Create a contextual data set: Cm = (xm,1, ym,1, ..., xm,N, ym,N), 
    where ym,i = fwm(xm,i) = wTm xm,i, and each wm ~ N(0d, Id).
    Each xm,i = alpha*v + beta*u + epsilon, where alpha ~ N(0, 1), beta ~ N(0, 1), 
    and epsilon = (epsilon_1, ..., epsilon_10)T , with epsilon_j ~ N(0, 1/100).
    """
    # Create vectors v, u
    v = np.array([np.cos(j * np.pi / 5) for j in range(1, i_size+1)])
    u = np.array([np.sin(j * np.pi / 5) for j in range(1, i_size+1)])

    # Create wm ~ N(0d, Id)
    wm = rng.normal(size=[i_size])

    # Create x, y, w
    xm = []
    ym = []
    for _ in range(c_size):
        

        # Create alpha ~ N(0, 1), beta ~ N(0, 1)
        alpha = rng.normal()
        beta = rng.normal()
        
        # Create epsilon ~ N(0, 1/100)
        epsilon = rng.normal(size=[i_size], scale = 1/10)

        # Compute x_m,i = alpha*v + beta*u + epsilon
        xm_i = alpha*v + beta*u + epsilon

        # Compute y_m,i = w^T x_m,i
        ym_i = np.matmul(wm, xm_i)

        xm.append(xm_i)
        ym.append(ym_i)

    # Convert to numpy arrays
    xm = np.array(xm)
    ym = np.array(ym)

    # Concatenate x_m,i and y_m,i
    Cm = np.concatenate([xm, ym[..., None]], axis = -1)

    return Cm, wm


def get_nonlinear_data(no_tasks, feature_size, no_examples):
    """
    Create a nonlinear regression data set: Cm = (xm,1, ym,1, ..., xm,N, ym,N), 
    where ym,i = fwm(xm,i) = wTm xm,i, and each wm ~ N(0d, Id).
    Each xm,i = alpha*v + beta*u + epsilon, where alpha ~ N(0, 1), beta ~ N(0, 1), 
    and epsilon = (epsilon_1, ..., epsilon_10)T , with epsilon_j ~ N(0, 1/100).
    """
    rng = np.random.default_rng()
    Cms, ws = [], []
    for _ in range(no_tasks):
        Cm, w = create_nonlinear_data(rng, no_examples, feature_size)
        Cms.append(Cm)
        ws.append(w)

    return np.array(Cms), np.array(ws)



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
    gd_plusplus = False,
):
    """
    Create linear regression gradient descent (or gd++) weights for linear self-attention
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
    param gd_plusplus: bool, if True, gd++ weights are created
    """
    if w_init is None:
        w_init = np.random.normal(size=[1, 1, feature_size]) * 0

    one = np.ones([feature_size + output_size])
    one_in_size = np.ones([feature_size])
    zero_out_size = np.zeros([output_size])
    one_out_size = np.ones([output_size])

    # Value matrix
    if gd_plusplus:
        value_upper_left = np.eye(feature_size)
        value_upper_right = np.zeros([feature_size, output_size])
        value_upper = np.concatenate([value_upper_left, value_upper_right], axis=1)
    else:
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
    if gd_plusplus:
        
        # projection_upper_left =  np.eye(feature_size) * (- gamma)
        # projection_upper_right = np.zeros([feature_size, output_size])
        # projection_upper_part = np.concatenate([projection_upper_left, projection_upper_right], axis=1)
        pass # how do i add gamma here?
    else:
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


######################################################################
## stock data
######################################################################

def get_stock_data(path, ticker):
    stocks = pd.read_csv(path)
    spy = stocks.loc[
    stocks["Ticker"] == ticker, ["DATE", "LAST", "OPEN", "LOW", "HIGH", "3M IMPLIED VOL"]
    ].dropna()
    spy.reset_index(drop=True, inplace=True)

    # Select all columns except 'DATE'
    cols = [col for col in spy.columns if col not in ["DATE", "3M IMPLIED VOL"]]

    # Apply the function to each element of the selected columns
    spy[cols] = spy[cols].apply(lambda x: np.log(x / 200))

    # Normalize the '3M IMPLIED VOL' column
    spy["3M IMPLIED VOL"] = spy["3M IMPLIED VOL"] / spy["3M IMPLIED VOL"].std()

    # create y
    spy["price_1d"] = spy["LAST"].shift(-1)
    spy["price_5d"] = spy["LAST"].shift(-5)
    spy["price_10d"] = spy["LAST"].shift(-10)
    spy["price_20d"] = spy["LAST"].shift(-20)
    spy["open_1d"] = spy["OPEN"].shift(-1)

    # divide the data into periods of 251 days
    n = 251
    spy["period"] = spy.index // n

    # for each period, separate the last 20 rows to avoid lookahead bias and use as eval data
    spy_val = spy.groupby("period").tail(20)
    spy = (
        spy.groupby("period")
        .apply(
            lambda x: x.iloc[:-20], include_groups=True
        )  # include groups to later create test query
        .reset_index(drop=True)
    )
    spy_val = spy_val.dropna()

    # Now let's fill the last entry of each period with 0.0
    columns = ["price_1d", "price_5d", "price_10d", "price_20d", "open_1d"]
    last_entry = spy.groupby("period").apply(
        lambda x: x.last_valid_index(), include_groups=False
    )
    # spy_targets = np.array(spy.loc[last_entry, :].drop(columns=["DATE", "period"]))
    spy_targets = np.array(spy.loc[last_entry, columns])
    spy.loc[last_entry, columns] = 0.0

    # Let's fill y data for spy_val with 0.0
    last_entry = spy_val.groupby("period").apply(
        lambda x: x.last_valid_index(), include_groups=False
    )
    spy_val_targets = np.array(spy_val.loc[last_entry, columns])
    spy_val.loc[last_entry, columns] = 0.0

    # We're going to treat each period as a context
    spy = spy.drop(columns=["DATE"])
    spy_val = spy_val.drop(columns=["DATE"])
    groups = [group.drop(columns=["period"]).values for _, group in spy.groupby("period")]
    groups_val = [
        group.drop(columns=["period"]).values for _, group in spy_val.groupby("period")
    ]
    spy_np = np.array(groups)
    spy_val_np = np.array(groups_val)

    # Define stock eval data
    spy_eval = (spy_val_np, spy_val_targets)
    # Define stock train data
    spy_train = (spy_np, spy_targets)
    # eval_targets = torch.tensor(spy_eval[1]).float()

    return spy_train, spy_eval