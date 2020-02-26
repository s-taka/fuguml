# -*- coding: utf-8 -*-

import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from fuguml.automl import AutoMLTable
import pprint as pp
import pandas as pd

famtbl = AutoMLTable({"struct": {"main": [("key", "key"), ("num1", "num"), ("cat1", "cat")],
                                       "tbl1": [("key", "key"), ("num2", "num"), ("cat2", "cat")]}})
famtbl.n_trials = 20
pp.pprint(famtbl.info)

main = pd.read_csv("tests/test1_main.tsv", sep="\t")
tbl1 = pd.read_csv("tests/test1_tbl1.tsv", sep="\t")
Xs = {"main": main, "tbl1": tbl1}
ret = famtbl.make_model(Xs, main[["obj"]])
pp.pprint(ret)

apply = pd.read_csv("tests/test1_apply.tsv", sep="\t")
Xs_apply = {"apply": apply, "tbl1": tbl1}

pred = famtbl.apply(Xs_apply)
pp.pprint(pred)

pred = famtbl.apply_cv(Xs_apply)
pp.pprint(pred)
