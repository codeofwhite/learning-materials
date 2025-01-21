import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simple_linear_regression(x, y):
    """Calculate linear regression parameters using simple formula."""


    w1 = 0.
    w0 = 0.
    return w1, w0


def matrix_linear_regression(x, y):
    """Calculate linear regression parameters using matrix operations."""

    w = [[0.], [0.]]
    return w[1][0], w[0][0]

if __name__ == "__main__":
    # Change the data path to the location where your dataset is stored.
    data_path = './dataset/Olympics/Olympics_data.csv'
    df_raw = pd.read_csv(data_path)
    x = df_raw['year']
    y = df_raw['time']

    # Calculate parameters using both functions
    w1_simple, w0_simple = simple_linear_regression(x, y)
    w1_matrix, w0_matrix = matrix_linear_regression(x, y)
    print("w1_simple:{}, w0_simple:{}".format(w1_simple, w0_simple))
    print("w1_matrix:{}, w0_matrix:{}".format(w1_matrix, w0_matrix))

    # Plotting the data and the regression lines


