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
pp.pprint(famtbl.info)
famtbl.n_trials = 10

main = pd.read_csv("tests/test1_main.tsv", sep="\t")
tbl1 = pd.read_csv("tests/test1_tbl1.tsv", sep="\t")

Xs = {"main": main, "tbl1": tbl1}
tbl_col_dic_list = famtbl.make_dic(Xs)
pp.pprint(tbl_col_dic_list)
features, features_info = famtbl.make_feature(Xs, tbl_col_dic_list)

pp.pprint(features.shape)
pp.pprint(features.todense())
pp.pprint(features_info)

model, cv_result, proba_model, proba_cv_result = famtbl.train(features, main[["obj"]], proba_calib=True)
pp.pprint(model)
pp.pprint(cv_result)
pp.pprint(proba_model)
pp.pprint(proba_cv_result)

importance, importance_dic = famtbl.get_importance(model, features_info)
pp.pprint(importance)
pp.pprint(importance_dic)

pp.pprint(famtbl.cv_result)

apply = pd.read_csv("tests/test1_apply.tsv", sep="\t")

Xs_apply = {"apply": apply, "tbl1": tbl1}
features_apply, features_apply_info = famtbl.make_feature_apply(Xs_apply, tbl_col_dic_list)
pp.pprint(features_apply.shape)
pp.pprint(features_apply.todense())
pp.pprint(features_apply_info)

pred = famtbl.predict(model, features_apply)
pp.pprint(pred)