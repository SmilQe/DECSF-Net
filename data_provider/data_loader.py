import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings("ignore")

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='weather.csv', target='Sal',
                 scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features    # M or MS or S
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 调整 df_raw 的列顺序，使 date 列排在最前，self.target 目标列排在最后，其余列位于中间
        # 此操作针对 M 和 MS 任务
        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]

        # 按照 7:2:1 的比例划分数据集
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # self.features == 'M' 多变量预测多变量 'MS' 多变量预测单变量
        # 'S' 单变量预测单变量(目标值预测目标值)
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 对数据进行归一化(标准化)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]    # 从 df_raw 中提取 date 列，并限制 [border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)    # 将 date 列的数据类型转换为 datetime 对象，对日期进行拆分

        if self.timeenc == 0:   # 根据 self.timeenc 的值，选择不同时间编码方式
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]     # x
        self.data_y = data[border1:border2]     # label
        self.data_stamp = data_stamp            # stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_MM(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M',
                 data_path='weather.csv', water_attrs=None, mete_attrs=None, target='OT', scale=True, timeenc=1, freq='h'):
        super(Dataset_MM, self).__init__()

        if water_attrs is None:
            self.water_attrs = ["Do", "Tem", "Ph", "Sal"]

        if mete_attrs is None:
            # self.mete_attrs = ["temp", "humi", "AtmosphericPressure",
            #               "windAngle", "windDirection", "WindSpeed",
            #               "WindDgree", "5MinRainfall", "hourlyRainfall", "daylyRainfall"]
            self.mete_attrs = ["temp", "humi", "AtmosphericPressure", "windAngle", "WindSpeed", "5MinRainfall"]
            # self.mete_attrs = ["temp", "humi", "AtmosphericPressure", "windDirection", "WindSpeed", "5MinRainfall"]

        # size [seq_water_len, seq_mete_len, label_len, pred_len]

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
            self.seq_mete_len = self.seq_len + self.pred_len
        else:
            self.seq_water_len = size[0]
            self.seq_mete_len = size[1]
            self.label_len = size[2]
            self.pred_len = size[3]

        # init
        assert flag in ['train', 'vali', 'test']
        type_map = {'train': 0, 'vali': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.target = target
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.water_scaler = StandardScaler()
        self.mete_scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
                df_raw.columns: ['date', (some water features), (some mete features)]
        '''

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        self.seq_len = self.seq_water_len if self.seq_water_len > self.seq_mete_len else self.seq_mete_len
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_water = df_raw[self.water_attrs]
        df_mete = df_raw[self.mete_attrs]

        if self.scale:
            train_water_data = df_water[border1s[0]: border2s[0]]   # 使用 train_water_data 的 mean 和 std，归一化数据集，然后再根据 train, vali, test 切分
            self.water_scaler.fit(train_water_data.values)
            water_data = self.water_scaler.transform(df_water.values)

            train_mete_data = df_mete[border1s[0]: border2s[0]]     # 使用 train_mete_data 的 mean 和 std，归一化数据集，然后再进行切分
            self.mete_scaler.fit(train_mete_data.values)
            mete_data = self.mete_scaler.transform(df_mete.values)

        else:
            water_data = df_water.values
            mete_data = df_mete.values

        df_stamp = df_raw[['date']][border1: border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.water_x = water_data[border1: border2]
        self.water_y = water_data[border1: border2]

        self.mete_x = mete_data[border1: border2]
        self.data_stamp = data_stamp

    def __len__(self):
        return len(self.water_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.water_scaler.inverse_transform(data)

    def __getitem__(self, index):
        w_begin = index
        w_end = w_begin + self.seq_water_len
        
        # [m_begin, m_end]: self.seq_mete_len
        m_begin = index
        m_end = m_begin + self.seq_mete_len
        
        # [m_begin, m_end]: self.pred_len
        # m_begin = w_end
        # m_end = m_begin + self.pred_len
        
        y_w_begin = w_end - self.label_len
        y_w_end = y_w_begin + self.label_len + self.pred_len

        water_seq_x = self.water_x[w_begin: w_end]
        water_mark_x = self.data_stamp[w_begin: w_end]

        mete_seq_x = self.mete_x[m_begin: m_end]
        mete_mark_x = self.data_stamp[m_begin: m_end]

        water_seq_y = self.water_y[y_w_begin: y_w_end]
        water_mark_y = self.data_stamp[y_w_begin: y_w_end]

        return water_seq_x, water_mark_x, mete_seq_x, mete_mark_x, water_seq_y, water_mark_y

if __name__ == '__main__':
    root_path = "../dataset"
    flag = 'train'
    size = [336, 336+96, 48, 96]        # [seq_water_len, seq_mete_len, label_len, pred_len]
    features = 'M'
    data_path = 'ffill_resampled.csv'

    dataset_MM = Dataset_MM(root_path, flag, size, features, data_path)

    data_loader = DataLoader(
        dataset=dataset_MM,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        drop_last=False,
    )


    print("Done")