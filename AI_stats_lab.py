import numpy as np


# -------------------------------------------------
# Question 1: Continuous pair on the unit square
# -------------------------------------------------

def joint_cdf_unit_square(x, y):

    if x <= 0 or y <= 0:
        return 0.0
    elif 0 < x < 1 and 0 < y < 1:
        return x * y
    elif 0 < x < 1 and y >= 1:
        return x
    elif x >= 1 and 0 < y < 1:
        return y
    else:  # x >= 1 and y >= 1
        return 1.0


def rectangle_probability(x1, x2, y1, y2):

    return (
        joint_cdf_unit_square(x2, y2)
        - joint_cdf_unit_square(x1, y2)
        - joint_cdf_unit_square(x2, y1)
        + joint_cdf_unit_square(x1, y1)
    )


def marginal_fx_unit_square(x):

    if 0 < x < 1:
        return 1.0
    else:
        return 0.0


def marginal_fy_unit_square(y):

    if 0 < y < 1:
        return 1.0
    else:
        return 0.0


# -------------------------------------------------
# Question 2: Joint PMF, marginals, independence
# -------------------------------------------------

def joint_pmf_heads(x, y):

    if x == 0 and y == 0:
        return 0.25
    elif x == 0 and y == 1:
        return 0.25
    elif x == 1 and y == 1:
        return 0.25
    elif x == 1 and y == 2:
        return 0.25
    else:
        return 0.0


def marginal_px_heads(x):

    # Sum over all y
    return (
        joint_pmf_heads(x, 0)
        + joint_pmf_heads(x, 1)
        + joint_pmf_heads(x, 2)
    )


def marginal_py_heads(y):

    # Sum over all x
    return (
        joint_pmf_heads(0, y)
        + joint_pmf_heads(1, y)
    )


def check_independence_heads():

    # Check if P(X,Y) = P(X)P(Y) for all combinations
    for x in [0, 1]:
        for y in [0, 1, 2]:
            joint = joint_pmf_heads(x, y)
            px = marginal_px_heads(x)
            py = marginal_py_heads(y)

            if not np.isclose(joint, px * py):
                return False

    return True
