from pysr import PySRRegressor
import torch
import numpy as np
from scipy.optimize import curve_fit
import warnings
import matplotlib.pyplot as plt
from ParticleGraph.utils import *
import matplotlib as mpl

def power_model(x, a, b):
    return a / (x ** b)


def linear_model(x, a, b):
    return a * x + b


def _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, idx):
    u = x[:, 3]
    v = x[:, 4]
    w = x[:, 5]

    laplacian = cc * x[:, idx]

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    return laplacian + a * uu + b * uv + c * uw + d * vv + e * vw + f * ww + g * u + h * v + i * w


def _aux_reaction_diffusion_L(x, cc, idx):
    u = x[:, 3]
    v = x[:, 4]
    w = x[:, 5]
    a = x[:, 6]
    b = x[:, 7]
    c = x[:, 8]
    d = x[:, 9]
    e = x[:, 10]
    f = x[:, 11]
    g = x[:, 12]
    h = x[:, 13]
    i = x[:, 14]

    laplacian = cc * x[:, idx]

    uu = u * u
    uv = u * v
    uw = u * w
    vv = v * v
    vw = v * w
    ww = w * w

    return laplacian + a * uu + b * uv + c * uw + d * vv + e * vw + f * ww + g * u + h * v + i * w


def reaction_diffusion_model(variable_name):
    match variable_name:
        case 'u':
            return lambda x, a, b, c, d, e, f, g, h, i, cc: _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, 0)
        case 'v':
            return lambda x, a, b, c, d, e, f, g, h, i, cc: _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, 1)
        case 'w':
            return lambda x, a, b, c, d, e, f, g, h, i, cc: _aux_reaction_diffusion(x, a, b, c, d, e, f, g, h, i, cc, 2)


def reaction_diffusion_model_L(variable_name):
    match variable_name:
        case 'u':
            return lambda x, cc: _aux_reaction_diffusion_L(x, cc, 0)
        case 'v':
            return lambda x, cc: _aux_reaction_diffusion_L(x, cc, 1)
        case 'w':
            return lambda x, cc: _aux_reaction_diffusion_L(x, cc, 2)

    def linear_fit(x_data=[], y_data=[], threshold=10):

        relative_error = np.abs(y_data - x_data) / x_data
        n_outliers = np.argwhere(relative_error < threshold)
        x_data_ = x_data[pos[:, 0]]
        y_data_ = y_data[pos[:, 0]]
        lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
        logger.info(' ')
        residuals = y_data_ - linear_model(x_data_, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return lin_fit, r_squared, relative_error, n_outliers, x_data, y_data


def linear_fit(x_data=[], y_data=[], threshold=10):

    if threshold ==-1:
        x_data_ = x_data
        y_data_ = y_data
        relative_error = -1
        not_outliers = -1

    else:
        relative_error = np.abs(y_data - x_data) / np.abs(x_data)
        not_outliers = np.argwhere(relative_error < threshold)
        x_data_ = x_data[not_outliers[:, 0]]
        y_data_ = y_data[not_outliers[:, 0]]

    lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
    residuals = y_data_ - linear_model(x_data_, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return lin_fit, r_squared, relative_error, not_outliers, x_data, y_data


def symbolic_regression(x,y):

    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    dataset = {}
    dataset['train_input'] = x[:, None]
    dataset['test_input'] = x[:, None]
    dataset['train_label'] = y[:, None]
    dataset['test_label'] = y[:, None]

    model_pysrr = PySRRegressor(
        niterations=30,  # < Increase me for better results
        binary_operators=["*", "+", "-", "/"],
        unary_operators=["square", "cube", "exp"],
        random_state=0,
        temp_equation_file=True
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_pysrr.fit(to_numpy(dataset["train_input"]), to_numpy(dataset["train_label"]))

    # print(model_pysrr)
    # print(model_pysrr.equations_)

    score = model_pysrr.equations_['score'][0:10]
    max_index = score.argmax()
    max_value = score[max_index]
    print(model_pysrr.sympy(max_index))

    return model_pysrr, max_index, max_value


def boids_model(x, a, b, c):
    xdiff = x[:, 0:2]
    vdiff = x[:, 2:4]
    r = np.concatenate((x[:, 4:5], x[:, 4:5]), axis=1)

    total = a*xdiff + b * vdiff - c * xdiff / r
    total = np.sqrt(total[:, 0] ** 2 + total[:, 1] ** 2)

    return total


def symbolic_regression_multi(x,y):

    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)

    y = 1*x[:,0] + 2*x[:,1] + 3*x[:,0]/(x[:,2]+1E-10)

    x[:, 0] = x[:, 0] / torch.std(x[:, 0])
    x[:, 1] = x[:, 1] / torch.std(x[:, 1])
    x[:, 2] = x[:, 2] / torch.std(x[:, 2])

    dataset = {}
    dataset['train_input'] = x
    dataset['test_input'] = x
    dataset['train_label'] = y
    dataset['test_label'] = y

    model_pysrr = PySRRegressor(
        niterations=300,  # < Increase me for better results
        binary_operators=["*", "+", "-"],
        unary_operators=["inv(x) = 1/x"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        constraints = {"mult": 1, "add": 1, "subb": 1, "inv": 1},
        nested_constraints={
            "*": {"inv": 1, "*": 0, "+": 1, "-": 1},
            "+": {"inv": 1, "*": 1, "+": 0, "-": 1},
            "-": {"inv": 1, "*": 1, "+": 1, "-": 0},
            "inv": {"inv":0, "*": 0, "+": 0, "-": 0}},
        select_k_features = 3,
        random_state=0,
        maxsize=100,
        weight_randomize=1,
        temp_equation_file=False,
        batching=True,
        model_selection='accuracy',
        batch_size=8,
        ncycles_per_iteration=5000,
        progress=True,
        verbosity=0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_pysrr.fit(to_numpy(dataset["train_input"]), to_numpy(dataset["train_label"]))
    print(model_pysrr)
    print(model_pysrr.equations_)
    y_ = model_pysrr.predict(to_numpy(x))

    fig, ax= fig_init(formatx='%.5f', formaty='%.5f')
    fmt = lambda x, pos: '{:.1f}e-4'.format((x) * 1e4, pos)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.scatter(to_numpy(y), y_, s=0.1, c='k',alpha=0.1)
    # plt.xlim([-1E-4, 1E-4])
    # plt.ylim([-1E-4, 1E-4])
    plt.tight_layout()

    score = model_pysrr.equations_['score'][0:10]
    max_index = score.argmax()
    max_value = score[max_index]
    print(model_pysrr.sympy(max_index))


    return model_pysrr, max_index, max_value