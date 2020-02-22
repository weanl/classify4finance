
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import json

from utils.utils import get_args
from utils.config import process_config
from utils.utils import accuracy_per_class

from data_loader.generator import DataGenerator

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

import datetime


class ResultRcoder:

    def __init__(self, idxs, labels, preds, matrix, total_acc, per_acc):
        self.idxs = idxs
        self.labels = labels
        self.preds = preds
        self.confusion_matrix = matrix
        self.total_acc = total_acc
        self.per_acc = per_acc


class ModelParas:

    def __init__(self, decision_function, kernel, c, gamma):
        self.decision_function = decision_function
        self.kernel = kernel
        self.c = c
        self.gamma = gamma


def run_mc_svm(config):
    data = DataGenerator(config)
    train_X, train_y, test_X, test_y = data.load(zscalar=True)
    train_idx, test_idx = [], []
    for c in data.c_label:
        train_idx.append(data.train_idxs[c])
        test_idx.append(data.test_idxs[c])
    # 记录训练集和测试集划分
    train_idx = np.concatenate(train_idx, axis=0)
    test_idx = np.concatenate(test_idx, axis=0)
    train_label, test_label = train_y, test_y
    # todo ovr and ovo
    param_grid = {
        'penalty': ['l1', 'l2'],
        "C": np.linspace(0.1, 10, 100),
        "multi_class": ["ovr"]
    }
    model = LogisticRegression(
        class_weight="balanced"
    )
    #
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=4,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(train_X, train_y)

    best_model = grid_search.best_estimator_
    best_params_ = grid_search.best_params_  # 记录模型超参数
    # print("best_clf\n", best_clf)
    print(best_params_)
    best_model.fit(train_X, train_y)
    #
    train_y_pred = best_model.predict(train_X)
    test_y_pred = best_model.predict(test_X)

    train_metric = accuracy_per_class(
        train_y,
        train_y_pred,
        labels=[1, 2, 3, 4, 5]
    )
    test_metric = accuracy_per_class(
        test_y,
        test_y_pred,
        labels=[1, 2, 3, 4, 5]
    )
    train_matrix = confusion_matrix(
        train_y,
        train_y_pred
    )
    test_matrix = confusion_matrix(
        test_y,
        test_y_pred
    )

    total_train_accuracy = np.trace(train_matrix) / train_y.shape[0]
    total_test_accuracy = np.trace(test_matrix) / test_y.shape[0]

    train_result_recoder = ResultRcoder(train_idx, train_label,
                                        train_y_pred, train_matrix, total_train_accuracy,
                                        train_metric)
    test_result_recoder = ResultRcoder(test_idx, test_label,
                                       test_y_pred, test_matrix, total_test_accuracy, test_metric)

    # case study
    data.load_case_file()
    data.case_preprocess()
    case_X, case_y = data.case_load()

    case_y_pred = best_model.predict(case_X)

    case_metric = accuracy_per_class(
        case_y,
        case_y_pred,
        labels=[1, 2, 3, 4, 5]
    )
    case_matrix = confusion_matrix(
        case_y,
        case_y_pred,
        labels=[1, 2, 3, 4, 5]
    )

    total_case_accuracy = np.trace(case_matrix) / case_y.shape[0]

    case_idx = np.arange(0, case_y.shape[0])
    case_result_recoder = ResultRcoder(case_idx, case_y, case_y_pred,
                                       case_matrix, total_case_accuracy, case_metric)
    return train_result_recoder, test_result_recoder, case_result_recoder, best_params_


if __name__ == "__main__":
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    if config.exp_name == "mc_lr":
        start_time = datetime.datetime.now()
        total_train_ac, total_test_ac, total_case_ac = [], [], []
        for r_turn in range(100):
            train_result_recoder, test_result_recoder, case_result_recoder, best_params_ = run_mc_svm(config)
            total_train_ac.append(train_result_recoder.total_acc)
            total_test_ac.append(test_result_recoder.total_acc)
            total_case_ac.append(case_result_recoder.total_acc)
            # 还需要记录 测试集 训练集 的划分，已经对应的真实标签，模型分类结果，案例的分类结果
            print("turn #", r_turn + 1, "train_ac:", train_result_recoder.total_acc, "test_ac:",
                  test_result_recoder.total_acc, "case_ac:", case_result_recoder.total_acc)
            # 将数据写入记录文件
            result_path = config.test_results + config.exp_name + '-' + str(config.train_rate) + '/' + \
                          str(round(train_result_recoder.total_acc, 3)) + '-' + \
                          str(round(test_result_recoder.total_acc, 3)) + '-' + \
                          str(round(case_result_recoder.total_acc, 3)) + '-' + \
                          "turn_" + str(r_turn) + ".txt"
            with open(result_path, "w", encoding="utf-8") as f:
                # 记录模型参数
                f.write("****** MODEL SUMMARY ******\n")
                json.dump(best_params_, f)
                f.write("\n")

                # ------
                # 训练结果
                f.write("****** TRAIN SUMMARY ******\n")
                train_num = train_result_recoder.idxs.shape[0]
                f.write("training num #" + str(train_num) + "\n")
                f.write("training confusion matrix:\n")
                for item in train_result_recoder.confusion_matrix:
                    item = [str(ele) for ele in item]
                    f.write("\t".join(item) + "\n")
                per_acc = train_result_recoder.per_acc
                per_acc = [str(ele) for ele in per_acc]
                f.write("training accuracy:\n")
                f.write("\t".join(per_acc) + "\n")
                f.write("total training accuracy #" + str(train_result_recoder.total_acc) + "\n")
                # 测试结果
                f.write("****** TEST SUMMARY ******\n")
                test_num = test_result_recoder.idxs.shape[0]
                f.write("testing num #" + str(test_num) + "\n")
                f.write("testing confusion matrix:\n")
                for item in test_result_recoder.confusion_matrix:
                    item = [str(ele) for ele in item]
                    f.write("\t".join(item) + "\n")
                per_acc = test_result_recoder.per_acc
                per_acc = [str(ele) for ele in per_acc]
                f.write("testing accuracy:\n")
                f.write("\t".join(per_acc) + "\n")
                f.write("total testing accuracy #" + str(test_result_recoder.total_acc) + "\n")
                # 案例结果
                f.write("****** CASE SUMMARY ******\n")
                case_num = case_result_recoder.idxs.shape[0]
                f.write("case num #" + str(case_num) + "\n")
                f.write("case confusion matrix:\n")
                for item in case_result_recoder.confusion_matrix:
                    item = [str(ele) for ele in item]
                    f.write("\t".join(item) + "\n")
                per_acc = case_result_recoder.per_acc
                per_acc = [str(ele) for ele in per_acc]
                f.write("case accuracy:\n")
                f.write("\t".join(per_acc) + "\n")
                f.write("total case accuracy #" + str(case_result_recoder.total_acc) + "\n")

                # ------
                # 训练细节
                f.write("****** TRAIN DETAIL ******\n")
                train_num = train_result_recoder.idxs.shape[0]
                for num in range(train_num):
                    f.write(
                        str(train_result_recoder.idxs[num]) + "\t" + str(train_result_recoder.labels[num]) + "\t" + str(
                            train_result_recoder.preds[num]) + "\n")
                # 测试结果
                f.write("****** TEST DETAIL ******\n")
                test_num = test_result_recoder.idxs.shape[0]
                for num in range(test_num):
                    f.write(
                        str(test_result_recoder.idxs[num]) + "\t" + str(test_result_recoder.labels[num]) + "\t" + str(
                            test_result_recoder.preds[num]) + "\n")
                # 案例结果
                f.write("****** CASE DETAIL ******\n")
                case_num = case_result_recoder.idxs.shape[0]
                for num in range(case_num):
                    f.write(
                        str(case_result_recoder.idxs[num]) + "\t" + str(case_result_recoder.labels[num]) + "\t" + str(
                            case_result_recoder.preds[num]) + "\n")

        end_time = datetime.datetime.now()
        total_train_ac = np.array(total_train_ac).mean()
        total_test_ac = np.array(total_test_ac).mean()
        total_case_ac = np.array(total_case_ac).mean()
        print("Finished\n", "total_train_ac:", total_train_ac, "\ntotal_test_ac:", total_test_ac, "\ntotal_case_ac:",
              total_case_ac)
        print("total time consuming #", end_time - start_time)
        result_log_path = config.test_results + config.exp_name + '-' + \
                          str(config.train_rate) + "/result_log" + ".txt"
        with open(result_log_path, "w", encoding="utf-8") as f:
            f.write("Finished\n")
            f.write("total training accuracy #" + str(total_train_ac) + "\n")
            f.write("total testing accuracy #" + str(total_test_ac) + "\n")
            f.write("total case accuracy #" + str(total_case_ac) + "\n")
            f.write("Start Time #" + str(start_time) + "\n")
            f.write("total time consumed #" + str(end_time - start_time) + "\n")
    elif config.exp_name == "mc_rf":
        pass
        # run_mc_rf2(config)
    else:
        pass
