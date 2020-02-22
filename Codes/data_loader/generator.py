import pandas as pd
import numpy as np

from utils.utils import split_index, split_index2


class DataGenerator:

    def __init__(self, config):
        self.config = config
        self.c_label = [1, 2, 3, 4, 5]
        self.load_file()
        self.preprocess()
        self.split()

    def load_file(self):
        self.data = {}
        for idx, name in enumerate(self.config.xlsx_files):
            # print(name)
            file_name = self.config.xlsx_file_path + name
            self.data[self.c_label[idx]] = pd.read_excel(
                file_name
            )

    def preprocess(self):
        # clean
        # select
        # imputation
        # label
        # aggregate
        self.data_agg = []
        for idx, label in enumerate(self.data.keys()):
            instance_num = self.data[label].shape[0]
            if label == 5:
                # label_truth = np.ones((instance_num, ), dtype=np.int)*(label-1)
                label_truth = np.ones((instance_num,), dtype=np.int) * label
            else:
                label_truth = np.ones((instance_num,), dtype=np.int) * label
            self.data[label]["truth"] = pd.Series(
                data=label_truth,
                index=self.data[label].index
            )
            # imputation
            self.data[label].fillna(self.data[label].median(), inplace=True)
            self.data_agg.append(self.data[label])
        self.data_agg = pd.concat(self.data_agg)

        self.company = self.data_agg.index
        self.number = self.data_agg.columns[0]
        self.attribs = self.data_agg.columns[1:-1]
        self.target = self.data_agg.columns[-1]

        self.attribs_mean = self.data_agg[self.attribs].mean()
        self.attribs_std = self.data_agg[self.attribs].std()
        # print(self.attribs_mean)
        # print(self.attribs_std)
        self.data_agg.to_excel(self.config.test_results + "data_agg.xlsx", )

    def split(self):
        train_rate = self.config.train_rate
        self.idxs = {}
        for idx, c in enumerate(self.c_label):  # (self.c_label[:-1]):
            self.idxs[c] = np.where(
                self.data_agg[self.target].values == c
            )[0]
        # print(self.idxs)

        self.train_idxs = {}
        self.test_idxs = {}
        for idx, c in enumerate(self.c_label):  # (self.c_label[:-1]):
            instance_num = self.idxs[c].shape[0]
            train_idx, test_idx = split_index(instance_num, train_rate)
            self.train_idxs[c] = self.idxs[c][train_idx]
            self.test_idxs[c] = self.idxs[c][test_idx]
        # print(self.train_idxs)
        # print(self.test_idxs)

    def load(self, zscalar=True):
        if zscalar == True:
            for idx, col in enumerate(self.attribs):
                self.data_agg[col] = (self.data_agg[col] - self.attribs_mean[col]) / self.attribs_std[col]
        else:
            pass
        # print(self.data_agg)
        self.train_X = []
        self.train_y = []
        self.test_X = []
        self.test_y = []
        values_X = self.data_agg[self.attribs].values
        values_y = self.data_agg[self.target].values
        for idx, c in enumerate(self.c_label):  # (self.c_label[:-1]):
            self.train_X.append(
                values_X[self.train_idxs[c]]
            )
            self.train_y.append(
                values_y[self.train_idxs[c]]
            )
            self.test_X.append(
                values_X[self.test_idxs[c]]
            )
            self.test_y.append(
                values_y[self.test_idxs[c]]
            )
        self.train_X = np.concatenate(self.train_X)
        self.train_y = np.concatenate(self.train_y)
        self.test_X = np.concatenate(self.test_X)
        self.test_y = np.concatenate(self.test_y)
        print("train_X:\t", self.train_X.shape)
        print("train_y:\t", self.train_y.shape)
        print("test_X:\t", self.test_X.shape)
        print("test_y:\t", self.test_y.shape)
        return self.train_X, self.train_y, self.test_X, self.test_y

    # 读取案例公司文件，存放在单个 excel 文件，且有 truth 列
    def load_case_file(self):
        # load case data
        file_name = self.config.xlsx_file_path + self.config.case_xlsx_file
        self.case_data = pd.read_excel(
            file_name
        )
        pass

    # 对案例公司数据进行预处理
    # 主要在前需要运行 self.preprocess() 获取均值和标准差
    def case_preprocess(self):
        # clean
        # select
        # imputation
        # label
        # aggregate

        # print(">>self.case_data:\n", self.case_data)
        #
        self.case_attribs = self.case_data.columns[1:-1]
        self.case_data[self.case_attribs] = self.case_data[self.case_attribs].fillna(
            self.case_data[self.case_attribs].median())
        self.case_company = self.case_data.index
        self.case_number = self.case_data.columns[0]
        self.case_target = self.data_agg.columns[-1]
        self.case_data.to_excel(self.config.test_results + self.config.case_xlsx_file)

        # print(">>case_company:\n", self.case_company)
        # print(">>case_number:\n", self.case_number)
        # print(">>case_target:\n", self.case_target)
        # print(">>case_attribs:\n", self.case_attribs)

        pass

    def case_load(self, zscalar=True):
        if zscalar == True:
            for idx, col in enumerate(self.case_attribs):
                self.case_data[col] = (self.case_data[col] - self.attribs_mean[col]) / self.attribs_std[col]
        else:
            pass

        case_X = self.case_data[self.case_attribs].values
        case_y = self.case_data[self.case_target].values
        print('case_X:\t', case_X.shape)
        print('case_y:\t', case_y.shape)

        return case_X, case_y
