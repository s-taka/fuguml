# -*- coding: utf-8 -*-

import CONSTANT


def make_key(val):
    try:
        ret = "{}".format(val)
        return ret
    except:
        return CONSTANT.UNKNOWN_KEY


def make_exist_celkey(row_num, col_num, col_dim):
    return row_num * col_dim + col_num
