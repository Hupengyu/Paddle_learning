# coding=utf-8
from __future__ import print_function


def get_max_np(np):
    x_min = np[0][0]
    x_max = np[0][0]
    y_min = np[1][1]
    y_max = np[1][1]
    for i in np:
        x_min = i[0] if (i[0] < x_min) else x_min
        x_max = i[0] if (i[0] >= x_max) else x_max
        y_min = i[1] if (i[1] < y_min) else y_min
        y_max = i[1] if (i[1] >= y_max) else y_max

    return y_min, y_max, x_min, x_max
