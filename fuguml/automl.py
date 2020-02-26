# -*- coding: utf-8 -*-

from . import CONSTANT

import scipy as sc
from scipy import sparse
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

import lightgbm as lgb
import numpy as np


def make_key(val):
    try:
        ret = "{}".format(val)
        return ret
    except:
        return CONSTANT.UNKNOWN_KEY


def make_exist_celkey(row_num, col_num, col_dim):
    return row_num * col_dim + col_num


def split_txt(txt):
    try:
        ary = txt.split(CONSTANT.TEXT_DELIMITER)
    except:
        ary = []
    return ary


class AutoMLTable:
    def __init__(self, info, key_col=CONSTANT.DATA_KEY_COL, keytime_col=CONSTANT.DATA_KEYTIME_COL,
                 main_tbl=CONSTANT.TRAIN_MAIN_TBL, apply_tbl=CONSTANT.APPLY_MAIN_TBL):
        # info = "struct": {table name: [col_name: operation]}
        self.info = info
        # tbl_col_dic = [(tbl_name, [(col: ({col_key: id}, max_id)]])]
        self.tbl_col_dic_list = []
        self.logs = []
        self.model = None
        self.cv_result = None
        self.proba_model = None
        self.proba_importances = None
        self.proba_cv_result = None
        self.importance = None
        self.n_trials = 100
        self.n_iterations = 800
        self.n_split = 3
        self.key_col = key_col
        self.keytime_col = keytime_col
        self.main_tbl = main_tbl
        self.apply_tbl = apply_tbl
        self.out_main_features = 10
        self.threashold = 0.0

    def logger(self, message):
        self.logs.append(message)

    def make_dic(self, Xs):
        tbl_struct = self.info[CONSTANT.INFO_STRUCT]
        tbl_col_dic_list = []
        for tbl in tbl_struct:
            tbl_cols = []
            for (col, ope) in tbl_struct[tbl]:
                if ope == CONSTANT.NUMERICAL_TYPE:
                    if tbl == self.main_tbl:
                        tbl_cols.append((col, {CONSTANT.NUM_ORG: 0}, 1, CONSTANT.NUMERICAL_TYPE_MAIN))
                    else:
                        tbl_cols.append((col, CONSTANT.NUM_FEATURES, CONSTANT.NUM_FEATURES_LEN, ope))
                elif ope == CONSTANT.CATEGORY_TYPE:
                    tmp = set()
                    for v in Xs[tbl][col].fillna(CONSTANT.CATEGORY_NULL).value_counts().index:
                        tmp.add(v)
                    if 0 == len(tmp):
                        continue
                    col_dic = {}
                    dic_idx = 0
                    for cv in tmp:
                        col_dic[make_key(cv)] = dic_idx
                        dic_idx += 1
                    tbl_cols.append((col, col_dic, dic_idx, ope))
                elif ope == CONSTANT.TEXT_TYPE:
                    tmp = set()
                    for txt in Xs[tbl][col].fillna(CONSTANT.CATEGORY_NULL).value_counts().index:
                        for v in split_txt(txt):
                            tmp.add(v)
                    if 0 == len(tmp):
                        continue
                    col_dic = {}
                    dic_idx = 0
                    for cv in tmp:
                        col_dic[make_key(cv)] = dic_idx
                        dic_idx += 1
                    tbl_cols.append((col, col_dic, dic_idx, ope))

            if len(tbl_cols) > 0:
                tbl_col_dic_list.append((tbl, tbl_cols))

        return tbl_col_dic_list

    def make_feature_apply(self, Xs, tbl_col_dic_list):
        return self.make_feature(Xs, tbl_col_dic_list, main_tbl_name=self.apply_tbl)

    def make_feature(self, Xs, tbl_col_dic_list, main_tbl_name=""):
        if len(main_tbl_name) == 0:
            main_tbl_name = self.main_tbl
        main_tbl = Xs[main_tbl_name]
        # {key: [line_num1, line_num2, ...]} for main table
        row_num = main_tbl.shape[0]
        key_row_nums = {}
        for i, v in enumerate(main_tbl[self.key_col]):
            if v in key_row_nums:
                key_row_nums[v].append(i)
            else:
                key_row_nums[v] = [i]

        # {line_num: key_time} for main table
        row_num_keytime = {}
        if self.keytime_col in main_tbl:
            for i, v in enumerate(main_tbl[self.keytime_col]):
                row_num_keytime[i] = v

        features, features_info = self.make_feature_main(main_tbl,
                                                         filter(lambda tpl: tpl[0] == self.main_tbl,
                                                                tbl_col_dic_list))

        for (tbl, col_list) in tbl_col_dic_list:
            if tbl == self.apply_tbl or tbl == self.main_tbl:
                continue
            for (col, col_dic, col_dim, col_ope) in col_list:
                target_df = Xs[tbl][[self.key_col, col]]
                if self.keytime_col in Xs[tbl]:
                    target_df = Xs[tbl][[self.key_col, col, self.keytime_col]]
                col_features, col_features_info = self.make_feature_cols(target_df, col_dic, col_dim, col_ope, row_num,
                                                                         key_row_nums, row_num_keytime)
                features.append(col_features)
                features_info.append(("{}.{}({})".format(tbl, col, col_ope), col_features_info))

        return sc.sparse.hstack(features).tocsc(), features_info

    def make_feature_main(self, main_tbl, tbl_col_dic):
        features = []
        features_info = []
        row_num = main_tbl.shape[0]
        dummy_col = "dummy"
        while dummy_col in main_tbl.columns:
            dummy_col += "_"
        main_tbl[dummy_col] = range(row_num)

        key_row_nums = {i: [i] for i in range(row_num)}
        min_date = 0
        if self.keytime_col in main_tbl:
            min_date = min(main_tbl[self.keytime_col])
        row_num_keytime = {i: min_date for i in range(row_num)}

        for (tbl, col_list) in tbl_col_dic:
            for (col, col_dic, col_dim, col_ope) in col_list:
                target_df = main_tbl[[dummy_col, col]]
                if self.keytime_col in main_tbl:
                    target_df = main_tbl[[dummy_col, col, self.keytime_col]]
                col_features, col_features_info = self.make_feature_cols(target_df, col_dic, col_dim, col_ope, row_num,
                                                                         key_row_nums, row_num_keytime)
                features.append(col_features)
                features_info.append(("{}.{}({})".format(self.main_tbl, col, col_ope), col_features_info))

        return features, features_info

    def make_feature_cols(self, df, col_dic, col_dim, ope, row_num, key_row_nums, row_num_keytime):
        mxcol = []
        mxrow = []
        mxdata = []
        list_idx = 0
        exist_cel = {}
        is_temporal = False
        if self.keytime_col in df.columns:
            is_temporal = True
        for row in df.values:
            row_key = row[0]
            row_val = row[1]
            row_keytime = 0
            if is_temporal:
                row_keytime = row[2]
            if row_key not in key_row_nums:
                continue

            if ope == CONSTANT.NUMERICAL_TYPE_MAIN:
                set_val = 0.0
                main_rows = key_row_nums[row_key]
                try:
                    set_val = float(row_val)
                except:
                    self.logger("cannot convert float({}) in KEY={}".format(row_val, row_key))
                for main_row in main_rows:
                    if not is_temporal or row_num_keytime[main_row] >= row_keytime:
                        exist_cel_key = make_exist_celkey(main_row, CONSTANT.COL_NUM_ORG, col_dim)
                        if exist_cel_key not in exist_cel:
                            mxcol.append(col_dic[CONSTANT.NUM_ORG])
                            mxrow.append(main_row)
                            mxdata.append(set_val)
                            exist_cel[exist_cel_key] = list_idx
                            list_idx += 1
            elif ope == CONSTANT.NUMERICAL_TYPE:
                set_val = 0.0
                main_rows = key_row_nums[row_key]
                try:
                    set_val = float(row_val)
                except:
                    self.logger("cannot convert float({}) in KEY={}".format(row_val, row_key))
                for main_row in main_rows:
                    if not is_temporal or row_num_keytime[main_row] >= row_keytime:
                        exist_cel_key = make_exist_celkey(main_row, CONSTANT.COL_NUM_MAX, col_dim)
                        if exist_cel_key not in exist_cel:
                            for col_key in CONSTANT.NUM_FEATURES.keys():
                                exist_cel_key = make_exist_celkey(main_row, col_dic[col_key], col_dim)
                                mxcol.append(col_dic[col_key])
                                mxrow.append(main_row)
                                if col_key == CONSTANT.NUM_LEN:
                                    mxdata.append(1.0)
                                else:
                                    mxdata.append(set_val)
                                exist_cel[exist_cel_key] = list_idx
                                list_idx += 1
                        else:
                            max_key = make_exist_celkey(main_row, CONSTANT.COL_NUM_MAX, col_dim)
                            min_key = make_exist_celkey(main_row, CONSTANT.COL_NUM_MIN, col_dim)
                            len_key = make_exist_celkey(main_row, CONSTANT.COL_NUM_LEN, col_dim)
                            sum_key = make_exist_celkey(main_row, CONSTANT.COL_NUM_SUM, col_dim)
                            avg_key = make_exist_celkey(main_row, CONSTANT.COL_NUM_AVG, col_dim)
                            mxdata[exist_cel[max_key]] = max(mxdata[exist_cel[max_key]], set_val)
                            mxdata[exist_cel[min_key]] = min(mxdata[exist_cel[min_key]], set_val)
                            mxdata[exist_cel[len_key]] += 1.0
                            mxdata[exist_cel[sum_key]] += set_val
                            mxdata[exist_cel[avg_key]] = mxdata[exist_cel[sum_key]] / mxdata[exist_cel[len_key]]
            elif ope == CONSTANT.CATEGORY_TYPE:
                set_key = make_key(row_val)
                if set_key not in col_dic:
                    continue
                set_colnum = col_dic[set_key]
                main_rows = key_row_nums[row_key]
                for main_row in main_rows:
                    if not is_temporal or row_num_keytime[main_row] >= row_keytime:
                        exist_cel_key = make_exist_celkey(main_row, set_colnum, col_dim)
                        if exist_cel_key not in exist_cel:
                            mxcol.append(set_colnum)
                            mxrow.append(main_row)
                            mxdata.append(1.0)
                            exist_cel[exist_cel_key] = list_idx
                            list_idx += 1
                        else:
                            mxdata[exist_cel[exist_cel_key]] += 1.0
            elif ope == CONSTANT.TEXT_TYPE:
                for word in split_txt(row_val):
                    set_key = make_key(word)
                    if set_key not in col_dic:
                        continue
                    set_colnum = col_dic[set_key]
                    main_rows = key_row_nums[row_key]
                    for main_row in main_rows:
                        if not is_temporal or row_num_keytime[main_row] >= row_keytime:
                            exist_cel_key = make_exist_celkey(main_row, set_colnum, col_dim)
                            if exist_cel_key not in exist_cel:
                                mxcol.append(set_colnum)
                                mxrow.append(main_row)
                                mxdata.append(1.0)
                                exist_cel[exist_cel_key] = list_idx
                                list_idx += 1
                            else:
                                mxdata[exist_cel[exist_cel_key]] += 1.0

        return sc.sparse.coo_matrix((mxdata, (mxrow, mxcol)), shape=(row_num, col_dim)), (col_dim, col_dic)

    def train(self, features, y, proba_calib=False):
        return make_lgbmodel(features, y, n_trials=self.n_trials, n_iterations=self.n_iterations,
                             n_split=self.n_split, proba_calib=proba_calib)

    def get_importance(self, model, features_info):
        feature_list_str = []
        for tpl in features_info:
            tbl_col = tpl[0]
            max_col = tpl[1][0]
            col_dic = tpl[1][1]
            rev_dic = {}
            for k in col_dic.keys():
                rev_dic[col_dic[k]] = k
            for i in range(max_col):
                if i in rev_dic:
                    feature_list_str.append("{}${}".format(tbl_col, rev_dic[i]))
                else:
                    feature_list_str.append("{}${}".format(tbl_col, "NONE"))
        importance = [(g, s, n, i) for i, (g, s, n) in enumerate(zip(model.feature_importance(importance_type='gain'),
                                                                     model.feature_importance(), feature_list_str))]
        importance_dic = {n: (g, s, i) for (g, s, n, i) in importance}
        return sorted(importance, reverse=True), importance_dic

    def get_importance_proba(self, model, features_info):
        importances = []
        feature_list_str = []
        for tpl in features_info:
            tbl_col = tpl[0]
            max_col = tpl[1][0]
            col_dic = tpl[1][1]
            rev_dic = {}
            for k in col_dic.keys():
                rev_dic[col_dic[k]] = k
            for i in range(max_col):
                if i in rev_dic:
                    feature_list_str.append("{}${}".format(tbl_col, rev_dic[i]))
                else:
                    feature_list_str.append("{}${}".format(tbl_col, "NONE"))
        for cc in model.calibrated_classifiers_:
            fi = cc.base_estimator.feature_importances_
            importance_dic = {n: (s, i) for i, (s, n) in enumerate(zip(fi, feature_list_str))}
            importances.append(importance_dic)

        return importances

    def predict(self, model, features):
        return model.predict(features)

    def predict_proba(self, model, features):
        return model.predict_proba(features)[:, 1]

    def choose_main_feature(self, features, importance, cv_importance):
        main_features = []
        imp_dic = {im[2]: im[0] for im in importance[0]}
        for imp in cv_importance:
            for im in imp[0]:
                im_w = im[0]
                im_name = im[2]
                imp_dic[im_name] += (1.0 / len(cv_importance)) * im_w
        imp_for_main_features = sorted([(imp_dic[k], k) for k in imp_dic], reverse=True)

        for i, (im_w, im_name) in zip(range(self.out_main_features), imp_for_main_features):
            im_num = importance[1][im_name][2]
            main_features.append((im_name, features.getcol(im_num)))

        return main_features

    def make_model(self, Xs, y):
        tbl_col_dic_list = self.make_dic(Xs)
        self.tbl_col_dic_list = tbl_col_dic_list
        features, features_info = self.make_feature(Xs, tbl_col_dic_list)
        model, cv_result, proba_model, proba_cv_result = self.train(features, y, proba_calib=True)
        self.model = model
        self.threashold = choose_threashold(features, y, model)
        self.proba_model = proba_model
        self.proba_importances = self.get_importance_proba(proba_model, features_info)
        self.proba_cv_result = proba_cv_result
        cv_importance = []
        proba_cv_importances = []
        for m, pm in zip(cv_result["models"], proba_cv_result["models"]):
            cv_importance.append(self.get_importance(m, features_info))
            proba_cv_importances.append( self.get_importance_proba(pm, features_info))
        self.cv_result = cv_result
        self.cv_result["importances"] = cv_importance
        self.proba_cv_result["importances"] = proba_cv_importances

        importance = self.get_importance(model, features_info)
        self.importance = importance
        main_features = self.choose_main_feature(features, importance, cv_importance)
        return self.cv_result["scores"], self.importance, self.proba_importances, self.cv_result, self.proba_cv_result, main_features

    def apply(self, Xs_apply, proba=False, out_main_feature=False):
        features_apply, features_apply_info = self.make_feature_apply(Xs_apply, self.tbl_col_dic_list)
        if proba:
            if out_main_feature:
                return self.predict(self.model, features_apply), \
                       self.predict_proba(self.proba_model, features_apply), \
                       self.choose_main_feature(features_apply, self.importance, self.cv_result["importances"])
            else:
                return self.predict(self.model, features_apply), self.predict_proba(self.proba_model, features_apply)
        else:
            if out_main_feature:
                return self.predict(self.model, features_apply), \
                       self.choose_main_feature(features_apply, self.importance, self.cv_result["importances"])
            else:
                return self.predict(self.model, features_apply)

    def apply_cv(self, Xs_apply, proba=False, out_main_feature=False):
        features_apply, features_apply_info = self.make_feature_apply(Xs_apply, self.tbl_col_dic_list)
        preds = np.zeros((features_apply.shape[0], len(self.cv_result["models"])))
        for i, m in enumerate(self.cv_result["models"]):
            p = self.predict(m, features_apply)
            preds[:, i] = p
        if not proba:
            if out_main_feature:
                return np.average(preds, axis=1), \
                       self.choose_main_feature(features_apply, self.importance, self.cv_result["importances"])
            else:
                return np.average(preds, axis=1)
        else:
            proba_preds = np.zeros((features_apply.shape[0], len(self.proba_cv_result["models"])))
            for i, m in enumerate(self.proba_cv_result["models"]):
                p = self.predict_proba(m, features_apply)
                proba_preds[:, i] = p
            if out_main_feature:
                return np.average(preds, axis=1), np.average(proba_preds, axis=1), \
                       self.choose_main_feature(features_apply, self.importance, self.cv_result["importances"])
            else:
                return np.average(preds, axis=1), np.average(proba_preds, axis=1)


def choose_threashold(X, y, model):
    ps, rs, ts = precision_recall_curve(y, model.predict(X))
    best_f = 0.0
    threashold = 0.0
    for idx, (p, r, t) in enumerate(zip(ps, rs, ts)):
        f = (2 * p * r) / (p + r)
        if f > best_f:
            best_f = f
            if idx + 1 < len(ts):
                threashold = (t + ts[idx + 1]) / 2.0
            else:
                threashold = t
    return threashold


def make_lgbmodel(features, y, n_trials=10, n_iterations=800, n_split=3, proba_calib=False):
    def obj(trial):
        train_x, test_x, train_y, test_y = train_test_split(features, y, test_size=0.5, random_state=1)
        lgb_train = lgb.Dataset(train_x, label=train_y)
        lgb_test = lgb.Dataset(test_x, label=test_y)

        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 0.5),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
            'max_depth': trial.suggest_int('max_depth', 1, 15),
            'num_leaves': trial.suggest_int('num_leaves', 10, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.2, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.2, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 30),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        }

        model1 = lgb.train(param, lgb_train, n_iterations, lgb_test, early_stopping_rounds=20, verbose_eval=0)
        score_test1 = model1.best_score["valid_0"][param["metric"]]
        best_itr = model1.best_iteration
        model2 = lgb.train(param, lgb_test, best_itr, lgb_train, verbose_eval=0)
        score_test2 = model2.best_score["valid_0"][param["metric"]]

        return (score_test1 + score_test2) / 2.0

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials)
    trial = study.best_trial
    base_param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt'}
    params = {**base_param, **trial.params}

    train_x, test_x, train_y, test_y = train_test_split(features, y, test_size=0.5, random_state=2)
    lgb_train = lgb.Dataset(train_x, label=train_y)
    lgb_test = lgb.Dataset(test_x, label=test_y)

    model1 = lgb.train(params, lgb_train, int(n_iterations * 1.5), lgb_test, early_stopping_rounds=20, verbose_eval=0)
    best_itr1 = model1.best_iteration
    model2 = lgb.train(params, lgb_test, int(n_iterations * 1.5), lgb_train, early_stopping_rounds=20, verbose_eval=0)
    best_itr2 = model2.best_iteration
    best_itr = int((best_itr1 + best_itr2) / 2.0)

    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=3)
    models = []
    scores = []
    threasholds = []
    np_y = np.array(y).ravel()
    preds = np.zeros(np_y.shape)
    preds_fold_num = np.zeros(np_y.shape)
    fold_num = 0
    proba_ret = {}
    if proba_calib:
        proba_ret = {"models": [], "scores": [], "preds": np.zeros(np_y.shape)}

    for train_index, test_index in skf.split(features, y):
        train_x, test_x = features[train_index], features[test_index]
        train_y, test_y = np_y[train_index], np_y[test_index]
        lgb_train = lgb.Dataset(train_x, label=train_y)
        cv_model = lgb.train(params, lgb_train, best_itr)
        cv_pred = cv_model.predict(test_x)
        cv_score = roc_auc_score(test_y, cv_pred)
        models.append(cv_model)
        scores.append(cv_score)
        threasholds.append(choose_threashold(train_x, train_y, cv_model))
        preds[test_index] = cv_pred
        preds_fold_num[test_index] = fold_num
        fold_num += 1
        if proba_calib:
            base_model = lgb.LGBMClassifier(**{**params, "n_estimators": best_itr, "importance_type": "gain"})
            proba_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

            proba_model.fit(train_x, train_y)
            proba_pred = proba_model.predict_proba(test_x)[:, 1]
            proba_ret["models"].append(proba_model)
            proba_ret["scores"].append(roc_auc_score(test_y, proba_pred))
            proba_ret["preds"][test_index] = proba_pred

    lgb_all = lgb.Dataset(features, label=y)
    if proba_calib:
        model = lgb.train(params, lgb_all, best_itr)
        cv_result = {"models": models, "scores": scores, "preds": preds,
                     "threasholds": threasholds, "preds_fold_num": preds_fold_num}
        base_model = lgb.LGBMClassifier(**{**params, "n_estimators": best_itr, "importance_type": "gain"})
        proba_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        proba_model.fit(features, y)
        return model, cv_result, proba_model, proba_ret

    else:
        return lgb.train(params, lgb_all, best_itr), {"models": models, "scores": scores,
                                                      "preds": preds, "preds_fold_num": preds_fold_num}
