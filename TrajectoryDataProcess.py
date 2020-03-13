# -*- coding: utf-8 -*-
# @Time    : 2019/7/12 17:12
# @Author  : xuzhifeng
# @File    : TrajectoryDataProcess.py
import os
import multiprocessing as mp
import pickle
import datetime
import time
import pandas as pd
import numpy as np
from FileConfig import Const

os_path = os.getcwd()

class TrajectoryDataProcess(object):
    def __init__(self, ITERATION):
        self.ITERATION = ITERATION

    def run(self):
        num = 0  # 索引值，遍历每一天的所有文件，用于保存第num个子文件
        full_files = []
        start_time = time.time()
        with open(r'./process_data/max_rows.pkl', 'rb') as f:
            max_rows = pickle.load(f)
        # pool = mp.Pool(15)  # 启用多进程并发执行每个文件
        # for root, dirc, filename in os.walk(r'J:\data_all\original_data\process_final'):  # 遍历每个文件
        #     for f in filename:
        #         full_files.append(os.path.join(root, f))
        # for path in full_files:
        #     day = str(path.split('\\')[-1]).strip("sz_.csv")
        #     index = int(day.split('_')[-1])
        #     # self.block_process(path, day, num, max_rows[index])
        #     pool.apply_async(self.block_process, args=(path, day, num, max_rows[index]))
        #     num = 0
        # pool.close()
        # pool.join()
        # over_time = time.time()
        # print("总共用时：", over_time - start_time)
        data = pd.read_csv(r'F:\data_all\original_data\process_final\sz_5_1.csv', nrows=10000,
                                infer_datetime_format = True, parse_dates = [3])
        self.process_data(data, '5_1', 1)
        # self.block_process(r'F:\data_all\original_data\process_final\sz_5_1.csv', num, max_rows[0])

    def delete_od_point(self, data):
        """
            剔除上下车数量小于5的轨迹点
        :param data:
        :return:存储上下车数量小于5的轨迹点所属车牌号的列表
        """
        grouped = data.groupby('taxi_id')
        id_list = []
        for n, f in grouped:
            taxi_id = f['taxi_id'].values[-1]
            l_1 = [j for j in f['status']]  # 用于存储 status 值的列表
            od_data = self.extract_od_point(l_1, f)
            if len(od_data) < 5:
                id_list.append(taxi_id)
        return id_list

    def statis_od_time(self, data, flag, df):
        """
              统计载客时长
                思路：拿到上下车点的时间戳数据 temp_up(上车时间) 、temp_off(下车时间) ，如果时间戳数据相等，则将其上
                     下车时间存入新表 dt 中，如果时间戳数据不相等，如：2015-05-01 00:51:09 2015-05-01 01:26:35
                     上下车时间不在一个时段内，要统计该时段的载客时间的话，则要把其分割成两行，一行表示 0 时的载客时间，另外
                     一行表示 1 时的载客时间。取出下车的时间戳，提取出该时间戳的年月日小时，
                     将其表示我字符串“2015-05-01 01:00:00”,然后再划分成
                     2015-05-01 00:51:09 2015-05-01 01:00:00 代表 0 时的载客时间段
                     2015-05-01 01:00:00 2015-05-01 01:26:35 代表 1 时的载客时间段
            :param data: 一辆出租车的数据
            :param flag: 统计各时段载客时间的下标值变量，当taxi_id + 1时候，相应的 flag+1
            :param df: 一张空的各时段的载客时间表
            :return:df: 统计完后，一辆车各时段载客时长的data
            """
        id = data['taxi_id'].values[-1]
        dt = pd.DataFrame(data=[[0, 0, 0]], columns=['taxi_id', 'pickup_datetime', 'dropoff_datetime'])
        num = 0  # 存储获取上下车准点信息的列表下标初始化变量
        new_data = pd.DataFrame()
        new_data = pd.concat([new_data, data])

        for i in range(new_data.shape[0]):
            new_data['pickup_datetime'] = pd.to_datetime(new_data['pickup_datetime'])
            new_data['dropoff_datetime'] = pd.to_datetime(new_data['dropoff_datetime'])
            temp_up = new_data.loc[i, 'pickup_datetime']
            temp_off = new_data.loc[i, 'dropoff_datetime']

            """
                获得下车时间O和上车时间U的hour差，记为D;
                U(i)= '{0}-{1}-{2} {3}:00:00'.format(U.year, U.month, U.day, U.hour + i)
                伪代码如下：
                IF O != U(0)
                    IF i == 1
                         dt['pickup_time'] = U(0)
                         dt['dropoff_time'] = U(i)
                    WHILE i>=1 and i < D
                         dt['pickup_time'] = U(i)
                         dt['dropoff_time'] = U(i+1)
                         i += 1
                    dt['pickup_time'] = U(i)
                    dt['dropoff_time'] = O
                ELSE
                    dt['pickup_time'] = U(0)
                    dt['dropoff_time'] = O
            """
            if temp_up.hour != temp_off.hour:
                i = 1
                d = temp_off.hour - temp_up.hour

                dt.loc[num, 'taxi_id'] = id
                dt.loc[num, 'pickup_datetime'] = temp_up
                dt.loc[num, 'dropoff_datetime'] = self.get_temptime(temp_up, i)

                while i >= 1 and i < d:
                    num += 1
                    dt.loc[num, 'taxi_id'] = id
                    dt.loc[num, 'pickup_datetime'] = self.get_temptime(temp_up, i)
                    dt.loc[num, 'dropoff_datetime'] = self.get_temptime(temp_up, i + 1)
                    i += 1

                num += 1
                dt.loc[num, 'taxi_id'] = id
                dt.loc[num, 'pickup_datetime'] = self.get_temptime(temp_up, i)
                dt.loc[num, 'dropoff_datetime'] = temp_off

                num = num + 1

            else:
                #  如果时间 hour 一样，则直接将其上下车时间戳插入新表 dt 中
                dt.loc[num, 'taxi_id'] = id
                dt.loc[num, 'pickup_datetime'] = temp_up
                dt.loc[num, 'dropoff_datetime'] = temp_off

            num = num + 1
        duration_data = dt
        #  计算载客时间，方法是：求下车点时间戳与下车点时间戳的时间间隔
        dt['trip_duration'] = (dt.loc[:, 'dropoff_datetime'] - dt.loc[:, 'pickup_datetime']) \
                                .apply(lambda x: x.total_seconds())

        #  按时间进行分组，然后分组统计每个时间段的载客时长
        dt['pickup_datetime'] = pd.to_datetime(dt['pickup_datetime'])
        dt['hour'] = dt['pickup_datetime'].dt.hour
        dt = dt.groupby('hour')
        taxi_id = new_data['taxi_id'].values[-1]

        #  分别将各时段不为空的载客时长存储到新表当中
        for hour, d in dt:
            df.loc[flag, 'taxi_id'] = taxi_id
            total_duration = d['trip_duration'].values.sum()
            df.loc[flag, '{}'.format(hour)] = total_duration

        return df, duration_data

    def statis_time_data(self, data, ps):
        """
            建立新的DataFrame, columns=['0',...,'23']
        :param data:
        :param ps:
        :param num:
        :param parameter:
        :return:
        """
        data = data.fillna(0)
        for j in range(0, 24):
            ps.loc[0, '{0}'.format(j)] = data['{0}'.format(j)].values.sum()
        return ps

    def statis_od_num(self, data, df_up, df_off, flag):
        """
            统计各时段O点和D点的数量
            思路：1、将 pickup_datetime 视为一次O点，即上车点；将 dropoff_datetime 视为一次D点，即下车点
                 2、前提条件就是： pickup_datetime 和 dropoff_datetime 都不能为空，为空计为0次

        :param data:
        :return: df: 统计完O点和D点数量的data
        """
        off_data = pd.DataFrame()
        up_data = pd.DataFrame()
        up_num = 0  # 统计上车总数量的变量
        off_num = 0  # 统计下车总数量的变量
        id = data['taxi_id'].values[-1]
        up_data = pd.concat([up_data, data], sort=False)
        up_data['pickup_hour'] = up_data['pickup_datetime'].dt.hour
        off_data = pd.concat([off_data, data[pd.notnull(data['dropoff_datetime'])]], sort=False)  # 如果最后一个数是Nat，则删除
        off_data['dropoff_hour'] = off_data['dropoff_datetime'].dt.hour

        o_data = up_data.groupby('pickup_hour')  # 统计上车点数量的 data_all
        d_data = off_data.groupby('dropoff_hour')  # 统计下车点数量的 data_all

        for hour, o in o_data:
            df_up.loc[flag, 'taxi_id'] = id
            df_up.loc[flag, '{}'.format(hour)] = o.shape[0]
            up_num = up_num + o.shape[0]

        df_up.loc[flag, 'total_num'] = up_num

        for hour, d in d_data:
            df_off.loc[flag, 'taxi_id'] = id
            df_off.loc[flag, '{}'.format(hour)] = d.shape[0]
            off_num = off_num + d.shape[0]

        df_off.loc[flag, 'total_num'] = off_num

        return df_up, df_off

    def statis_num_data(self, data, ps):
        """
            建立新的DataFrame, columns=['total_num','0',...,'23']
        :param data:
        :param ps:
        :param num:
        :param parameter:
        :return:
        """
        data = data.fillna(0)
        ps.loc[0, 'total_num'] = data['total_num'].values.sum()

        for j in range(0, 24):
            ps.loc[0, '{0}'.format(j)] = data['{0}'.format(j)].values.sum()
        return ps

    def statis_duration_od(self, od_data):
        """
             统计一辆车的OD时间和各时段的载客总时间
             调用了 statis_od_time()函数、 statis_time_data()函数
         :param od_data: 一个part-m-x文件数据
         :return: 载客信息表、各时段载客时间表、上车点表，下车点表
         """
        # 1、统计载客信息和载客时间
        od_data.dropna(how='any', inplace=True)  # 先删除 有上车点，无下车点的数据
        new_data = od_data.groupby('taxi_id')
        duration_time = pd.DataFrame()  # 存储载客时长
        zaike_info = pd.DataFrame()  # 存储返回的载客表
        flag = 0  # 存储24小时各时段 trip_duration 的列表下标初始化变量
        col_time = ['{}'.format(i) for i in range(0, 24)]
        cow_time = ['{}'.format(i) for i in range(0, 24)]
        col_time.insert(0, 'taxi_id')
        df = pd.DataFrame([[0] * 25], columns=col_time)
        df_time = pd.DataFrame([[0] * 24], columns=cow_time)
        for index, d in new_data:
            d.reset_index(drop=True, inplace=True)
            duration_time, duration_data = self.statis_od_time(d, flag, df)  # 返回一辆车的载客时间和各时段的载客时间
            zaike_info = pd.concat([zaike_info, duration_data])
            flag = flag + 1
        zaike_info = zaike_info.reset_index(drop=True)  # 得到 part-m-0 一个文件的数据
        df_time = self.statis_time_data(duration_time, df_time)  # 得到 part-m-0 一个文件的总时间

        # 2、统计up、off点的数量
        temp = 0  # 存储24小时各时段 up、off点的列表下标初始化变量
        up = None
        off = None
        col_num = ['{}'.format(i) for i in range(0, 24)]
        cow_num = ['{}'.format(i) for i in range(0, 24)]
        col_num.insert(0, 'taxi_id')
        cow_num.insert(0, 'total_num')
        col_num.insert(1, 'total_num')
        df_up = pd.DataFrame([[0] * 26], columns=col_num)
        df_off = pd.DataFrame([[0] * 26], columns=col_num)
        ps_up = pd.DataFrame([[0] * 25], columns=cow_num)
        ps_off = pd.DataFrame([[0] * 25], columns=cow_num)
        new_data = od_data.groupby('taxi_id')
        for index, d in new_data:
            d.reset_index(drop=True, inplace=True)
            up, off = self.statis_od_num(d, df_up, df_off, temp)
            temp = temp + 1
        ps_up = self.statis_num_data(up, ps_up)
        ps_off = self.statis_num_data(off, ps_off)

        return zaike_info, df_time, ps_up, ps_off

    def extract_carray_time(self, data):
        """
            提取一辆出租车的运营时间
        :param data:
        :return:车辆运营表
        """
        num = 0
        df = pd.DataFrame(data=[[0, 0, 0]], columns=['taxi_id', 'start_time', 'over_time'])
        data = data[['taxi_id', 'date_time', 'hour']]
        data = data.groupby('taxi_id')
        for index, dt in data:
            dt = dt.groupby('hour')
            for h, d in dt:
                d = d.reset_index(drop=True)
                taxi_id = d.loc[0, 'taxi_id']
                df.loc[num, 'taxi_id'] = taxi_id
                df.loc[num, 'start_time'] = d['date_time'].iloc[0]
                df.loc[num, 'over_time'] = d['date_time'].iloc[-1]
                num = num + 1
        return df

    def extract_final_data(self, d_1, d_2):
        """
            目标：获得一辆车在各个时段的真实运营时间
            思路：1、对车牌进行分组，在按时间段分组，分别统计各个时段的载客时间，运营时间
            问题：1、如果上车时间和下车时间在同一个时段内，载客时间就是该时段内上下车点时间间隔的总和；
                    运营总时间就是该时段的最后一个记录点减去第一个记录点的时间差
                 2、如果上车时间和下车时间不在同一个时段内，比如上车时间是08:12:32,下车时间是09:56:44，
                    那这个时间就要分成2段来统计，08:12:32-09:00:00，09:00:00-09:56:44,然后再计算
                    每个时段的载客时间和总时间
                 3、进行分段统计之后，该时段的载客总时间也要相应的改变，如：上面示例中，如果8点这个时刻的最后
                    时间变成了09:00:00，这个时刻是处于载客状态的，但是该时段的最后一个记录时刻为8:54:23，那么
                    我们简单做个对比，9:00:00 > 08:54:23, 算运营时间的时候，用9:00:00 - 第一个记录点时刻
                    即可，其他时间段以此类推。
        :param d_1: 一辆车载客的实际上下车时间
        :param d_2: 一辆车GPS记录仪上记录的实际时间
        :return: 一辆车运营的总时间
        """
        d_3 = pd.DataFrame()
        d_1 = d_1.reset_index(drop=True)
        d_2 = d_2.reset_index(drop=True)
        d_1['pickup_datetime'] = pd.to_datetime(d_1['pickup_datetime'])
        d_1['hour'] = d_1['pickup_datetime'].dt.hour
        d_1.drop_duplicates(subset=['hour'], keep='last', inplace=True)
        d_1.reset_index(drop=True, inplace=True)

        d_2['start_time'] = pd.to_datetime(d_2['start_time'])
        d_2['hour'] = d_2['start_time'].dt.hour
        d_3 = pd.concat([d_3, d_2])
        for i1, h1 in enumerate(d_1['hour']):
            for i2, h2 in enumerate(d_2['hour']):
                if h1 == h2:
                    if d_1.loc[i1, 'dropoff_datetime'] > d_2.loc[i2, 'over_time']:
                        d_3.loc[i2, 'over_time'] = d_1.loc[i1, 'dropoff_datetime']
                        d_3.loc[i2 + 1, 'start_time'] = d_1.loc[i1, 'dropoff_datetime']
        return d_3

    def compare_data(self, zaike_info, carray_data):
        """
            1、分别获得载客和运营的记录表，然后循环两表中相同的taxi_id，将得到的载客表d_1、运营表d_2
               传入到 compare_data(d_1, d_2) 函数中
            2、将运营表赋给新表d_3，对比d_1和d_2中的dropoff_datetime、over_time
               如果 over_time > dropoff_datetime, 则存放到新表中的 over_time 下相应的位置，否则不改变
        :param day:
        :return: None
        """
        final_data = pd.DataFrame()
        zaike_g = zaike_info.groupby(['taxi_id'])
        total_g = carray_data.groupby(['taxi_id'])
        for zaike, total in zip(zaike_g, total_g):
            if zaike[0] == total[0]:
                final_data = pd.concat([final_data, self.extract_final_data(zaike[1], total[1])])
                final_data = final_data.reset_index(drop=True)

        return final_data

    def sort_datetime(self, data):
        """
            对数据集中的taxi_id分组，并对时间序列 date_time 按升序进行排序
        :param data:
        :return: 排序好的data
        """
        grouped = data.sort_values('date_time').groupby('taxi_id')
        df = pd.DataFrame()
        for i, j in grouped:
            df = df.append(j)
        return df

    def block_process(self, path,day, num, max_rows):
        """
             分块处理数据
        :param path: 每一天的csv文件的地址目录
        :param day: 第几天的数据
        :param num:  第num个文件
        :param max_rows: 该文件的最大行数
        :return:
        """
        temp_dt = pd.read_csv(path, iterator=True, infer_datetime_format=True, parse_dates=[3])
        last_dt = pd.read_csv(path, iterator=True, infer_datetime_format=True, parse_dates=[3])
        loop = True
        chunks = []
        total_taxi = 0  # 出租车总数量
        empty_taxi = 0  # 空驶出租车数量
        temp_1 = 100000  # 设置查询列表下标
        temp_2 = 0  # 临时变量
        temp_3 = 0  # 临时变量
        temp_list = []  # 存储基于taxi_id的chunkSize大小
        while loop:
            try:
                t = temp_dt.get_chunk(self.ITERATION)
                chunks.append(t)
                da = pd.concat(chunks, ignore_index=True)
                m = da.loc[temp_1 - 3, 'taxi_id']
                n = da.loc[temp_1 - 2, 'taxi_id']
                z = da.loc[temp_1 - 1, 'taxi_id']
                if m != n and n == z:
                    temp_1 = temp_1 - 2
                elif m == n and n != z:
                    temp_1 = temp_1 - 1
                elif m == n and n == z:
                    f = temp_1 - 3
                    while True:
                        if m != da.loc[f - 1, 'taxi_id']:
                            break
                        f = f - 1
                    temp_1 = f
                else:
                    pass
                w = temp_1
                chunksize = temp_1 - temp_3
                temp_1 = temp_1 + self.ITERATION
                if max_rows - temp_1 > self.ITERATION - 1:
                    temp_3 = temp_3 + chunksize
                else:
                    temp_2 = max_rows - w
                    loop = False
                temp_list.append(chunksize)
                if temp_2 != 0:
                    temp_list.append(temp_2)
                c = last_dt.get_chunk(temp_list.pop())
                c = self.sort_datetime(c)
                total, empty = self.process_data(c, day, num)  # 将分块得到的数据传入进行预处理
                total_taxi += total
                empty_taxi += empty
                num += 1
            except StopIteration:
                loop = False
                print("Iteration is stopped.")

        content = "出租车总数量：{}".format(total_taxi) + "  空驶出租车的数量为：{}".format(empty_taxi)
        os.chdir(os_path)
        os.chdir(r'J:/data_all/process_data/statis_taxinum/{}'.format(day))
        s = open('{}.txt'.format(day), 'w', encoding='utf-8')
        s.write(content)
        print(day, "的数据处理完毕.")
        os.chdir(os_path)

    def get_trajectory_info(self, od_list, time, lat, lng, speed, direction, data, track_num):
        """
            获得出租车载客状态的完整轨迹数据
        :param od_list:
        :param time:
        :param lat:
        :param lng:
        :param data:
        :param track_num:
        :return:
        """
        taxi_id = data.loc[0, 'taxi_id']
        num = 0
        flag = True
        dt = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0, 0]],
                          columns=['track_id', 'taxi_id', 'date_time', 'lat', 'lng', 'speed', 'direction'])
        for j in od_list:
            if j[-1] == None:
                pass
            else:
                track_count = j[-1] - j[0] + 1  # 轨迹点的数量，小于10条轨迹点的数据都给予剔除
                if track_count > 0:
                    for m in range(j[0], j[-1] + 1):
                        dt.loc[num, 'track_id'] = track_num
                        dt.loc[num, 'date_time'] = time[m]
                        dt.loc[num, 'lat'] = lat[m]
                        dt.loc[num, 'lng'] = lng[m]
                        dt.loc[num, 'speed'] = speed[m]
                        dt.loc[num, 'direction'] = direction[m]
                        num = num + 1
                    track_num = track_num + 1
                    dt.loc[:, 'taxi_id'] = taxi_id
                    dt['date_time'] = pd.to_datetime(dt['date_time'])
                    dt['track_id'] = dt['track_id'].apply(lambda x: int(x))
                else:
                    flag = False
        return dt, track_num, flag

    def transform_to_mapmatch(self, data):
        """
        将提取轨迹后的数据进行转换，转换成地图匹配要求的数据输入格式
        :param data: 提取轨迹后的数据
        :return:
        """
        new_data = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0, 0, 0]],
                                columns=['uuid', 'track_id', 'log_time', 'v', 'car_id', 'lng', 'lat', 'direction'])
        num = 0
        max_num = data.shape[0]
        for i in range(1, max_num + 1):
            new_data.loc[num, 'uuid'] = int(i)
            num += 1
        new_data['uuid'] = new_data['uuid'].astype(np.int32)
        new_data.loc[:, 'car_id'] = data['taxi_id'].apply(lambda x: str(x).replace("粤", 'Y'))
        new_data.loc[:, 'v'] = data['speed'].astype(np.int32)
        new_data.loc[:, 'direction'] = data['direction'].astype(np.int32)
        new_data['log_time'] = data['date_time']
        new_data['track_id'] = data['track_id']
        new_data['lng'] = data['lng']
        new_data['lat'] = data['lat']

        return new_data

    def calculation_od_status(self, list_data):
        """
            返回载客状态连续为1和不连续1的列表
        :param list_data:
        :return:
        """
        t_1 = []  # 存储不连续1的index列表
        t_2 = []  # 临时存储连续1的index列表
        t_3 = []  # 存储连续1的index列表
        """ 计算不连续1的个数 """
        #  列表第一个为1，且第二个不为1,可看做不连续的1
        if (list_data[0] == 1) & (list_data[1] != 1):
            t_1.append(0)

        #  列表从第1个到倒数第2个循环
        for i in range(len(list_data) - 2):
            if (list_data[i + 1] == 1) & (list_data[i + 1] != list_data[i + 2]) & (list_data[i + 1] != list_data[i]):
                t_1.append(i + 1)
            #  两两比较，只要满足当前数为1，但前后两个数不为1，就把该数加入新的列表
            if (i == len(list_data) - 3) & (list_data[i + 2] == 1) & (list_data[i + 1] != 1):
                t_1.append(i + 2)

        """计算连续1的个数 """
        n = 0
        for j in range(0, len(list_data) - 1):
            if (list_data[j] == 1) & (list_data[j] == list_data[j + 1]):
                t_2.append(j)
                n = n + 1
            if (list_data[j + 1] == 0) | (j == len(list_data) - 2):
                if j != len(list_data) - 2:
                    if n >= 1:
                        t_2.append(j)
                        t_3.append(t_2)
                        n = 0
                        t_2 = []
                else:
                    if (j == len(list_data) - 2) & (list_data[j + 1] == 0):
                        if n >= 1:
                            t_2.append(j)
                            t_3.append(t_2)
                            n = 0
                            t_2 = []

                    elif (j == len(list_data) - 2) & (list_data[j + 1] == 1):
                        if n >= 1:
                            t_2.append(j + 1)
                            t_3.append(t_2)
                            n = 0
                            t_2 = []
        t_1 = [x for x in t_1 if x != []]
        t_3 = [x for x in t_3 if x != []]
        return t_1, t_3

    def extract_od_point(self, list_point, data):
        """
            提取上下车OD点算法
        :param list_point:
        :param data:
        :return:
        """
        data = pd.DataFrame(data).reset_index(drop=True)
        t_1, t_3 = self.calculation_od_status(list_point)
        max_index = data.shape[0] - 1  # 最大列表下标值
        od_list = []
        temp_t3 = None
        temp_t1 = None

        #  判断 t_3 的最后一个元素是否是data的最大下标值
        if len(t_3) > 1:
            temp_t_3 = t_3[-1]
            if temp_t_3[-1] == max_index:
                # 如果是最大下标值，则其无下车点(在次日)，将上车点存入列表中
                temp_t3 = [temp_t_3[0], None]
            else:
                pass
        elif len(t_3) == 1:
            if t_3[-1] == max_index:
                # 如果是最大下标值，则其无下车点(在次日)，将上车点存入列表中
                temp_t3 = [t_3[0], None]
            else:
                pass
        else:
            pass

        #  判断 t_1 的最后一个元素是否是data的最大下标值
        if len(t_1) > 1:
            temp_t_1 = t_1[-1]
            if temp_t_1 == max_index:
                # 如果是最大下标值，则其无下车点(在次日)，将上车点存入列表中
                temp_t1 = [temp_t_1, None]
        elif len(t_1) == 1:
            temp_t_1 = t_1[-1]
            if temp_t_1 == max_index:
                # 如果是最大下标值，则其无下车点(在次日)，将上车点存入列表中
                temp_t1 = [temp_t_1, None]
            else:
                pass
        else:
            pass

        #  将 t_1、t_3 两个列表进行合并
        for i in t_1:
            t_3.append([i])

        #  提取OD点（列表中第一个1即为O点，最后一个1的后一个数如为1即为D点）
        for i in t_3:
            o_point = i[0]
            last_i = i.pop() + 1
            if last_i < max_index:
                j = data.loc[last_i, 'status']
                if j == 0:
                    d_point = last_i
                    od_list.append([o_point, d_point])

        #  如果temp_t1、temp_t3不为空，即data最大下标值代表的状态值为1，则将其存入列表当中
        if temp_t1 is not None:
            od_list.append(temp_t1)
        elif temp_t3 is not None:
            od_list.append(temp_t3)

        #  对得到的列表按升序进行排序
        od_list = sorted(od_list, key=lambda x: x[0])
        return od_list

    def haversine_array(self, lat1, lng1, lat2, lng2):
        """
            计算地理空间距离
        :param lat1:
        :param lng1:
        :param lat2:
        :param lng2:
        :return:
        """
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    def get_temptime(self, temp_up, i):
        """
            格式化用于替换的时间戳
        :param temp_up:
        :param i:
        :return:
        """
        timestring = '{0}-{1}-{2} {3}:00:00'.format(temp_up.year, temp_up.month, temp_up.day, temp_up.hour + i)
        temp_date = time.mktime(time.strptime(timestring, '%Y-%m-%d %H:%M:%S'))
        temp = datetime.datetime.fromtimestamp(temp_date)
        return temp

    def get_distance(self, lat_in, lon_in):
        """
            输入经纬度信息，返回地理空间距离
        :param lat_in:
        :param lon_in:
        :return: 地理空间距离
        """
        lat = np.array(lat_in)
        lon = np.array(lon_in)
        lat *= np.pi / 180
        lon *= np.pi / 180
        lon1 = lon[0:-1]
        lon2 = lon[1:]
        lat1 = lat[0:-1]
        lat2 = lat[1:]
        x = (lon2 - lon1) * np.cos((lat1 + lat2) / 2)
        y = lat2 - lat1
        d = np.sqrt(x ** 2 + y ** 2) * 6371004
        return d

    def get_distance_intervals(self, data):
        """
            计算2个经纬度点之间的距离间隔
        :param data:
        :return: 距离间隔
        """
        i = 1
        for m, n in data.groupby('taxi_id'):
            n = pd.DataFrame(n)
            for distance_intervals in self.get_distance(n['lat'].values, n['lng'].values):
                data.loc[i, 'distance_intervals'] = distance_intervals
                i = i + 1
            i = i + 1
        return data['distance_intervals']

    def get_time_intervals(self, data):
        """
            计算2个经纬度点之间的时间间隔
        :param data:
        :return: 时间间隔
        """
        i = 0
        for m, n in data.groupby('taxi_id'):
            for time_intervals in n['date_time'].diff():
                time_intervals /= np.timedelta64(1, 's')
                data.loc[i, 'time_intervals'] = time_intervals
                i = i + 1
        return data['time_intervals']

    def get_od_point(self, data):
        """
        对地图匹配完成后的轨迹点，提取旅行的起点和终点
        :param data:
        :return:
        """
        data = data.groupby('track_id')
        new_data = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0, 0]], columns=['taxi_id', 'pickup_datetime', 'dropoff_datetime',
                                                                       'pickup_longitude', 'pickup_latitude',
                                                                       'dropoff_longitude', 'dropoff_latitude'])
        num = 0
        for track_id, dt in data:
            new_data.loc[num, 'taxi_id'] = str(dt['car_id'].values[0])[0].replace('Y', "粤") + str(
                dt['car_id'].values[0])[1:]
            time1 = pd.to_datetime(dt['log_time'].values[0])
            time2 = pd.to_datetime(dt['log_time'].values[-1])
            lng1 = dt['lng'].values[0]
            lat1 = dt['lat'].values[0]
            lng2 = dt['lng'].values[-1]
            lat2 = dt['lat'].values[-1]
            new_data.loc[num, 'pickup_datetime'] = time1
            new_data.loc[num, 'dropoff_datetime'] = time2
            new_data.loc[num, 'pickup_longitude'] = lng1
            new_data.loc[num, 'pickup_latitude'] = lat1
            new_data.loc[num, 'dropoff_longitude'] = lng2
            new_data.loc[num, 'dropoff_latitude'] = lat2
            travel_time = (time2 - time1).seconds
            distance = self.haversine_array(lat1, lng1, lat2, lng2)
            new_data.loc[num, 'distance'] = distance
            new_data.loc[num, 'travel_time'] = travel_time
            new_data.loc[num, 'average_time'] = 3600 * distance / travel_time
            num += 1
        return new_data

    def od_extract_main(self, data, track_num):
        """
            计算载客的上下车时间、经纬度的函数入口

            提取轨迹编号
        :param data:
        :param track_num:
        :return:
        """
        grouped = data.groupby('taxi_id')
        od_track = pd.DataFrame()

        for n, f in grouped:
            f = f.reset_index(drop=True)
            l_1 = [j for j in f['status']]  # 用于存储 status 值的列表
            l_2 = [i for i in f['date_time']]  # 用于存储 date_time 值的列表
            l_3 = [p for p in f['lat']]  # 用于存储 lat 值的列表
            l_4 = [q for q in f['lng']]  # 用于存储 lng 值的列表
            l_5 = [h for h in f['speed']]  # 用于存储 speed 值的列表
            l_6 = [m for m in f['direction']]  # 用于存储 direction 值的列表
            od_dt, track_num, flag = self.get_trajectory_info(self.extract_od_point(l_1, f), l_2, l_3, l_4, l_5, l_6, f, track_num)
            if od_dt.shape[0] > 1:
                od_track = pd.concat([od_track, od_dt], ignore_index=True)
            if flag:
                track_num += 1
        od_track = od_track.replace(0, np.nan)
        od_track.dropna(inplace=True)
        match_data = self.transform_to_mapmatch(od_track)  # 转换成地图匹配时需要的数据输入
        od_data = self.get_od_point(match_data)  # 提取轨迹的上下车点
        return od_data, match_data

    def carry_passengers_static(self, list_all, data):
        """
            返回一辆出租车载客时长和载客次数的DataFrame
            统计载客次数和载客时长，运营状态中，每存在一个有客段就代表一次载客，即连续1的数据，代表居民一次出行
        :param list_all:
        :param data:
        :return:
        """
        long = []
        times = []
        taxi_num = data['taxi_id']
        f = [i for i in range(0, 24)]
        df = pd.DataFrame()
        df['hour'] = f
        df['taxi_id'] = 0
        df['time_intervals'] = 0
        for list_first, time in list_all:
            t_3 = []  # 存储连续1的index列表
            #  判断list列表长度是否大于3
            if len(list_first) < 3:
                if len(list_first) == 0:
                    t_1 = [0]
                elif len(list_first) == 1:
                    if list_first[0] == 0:
                        t_1 = []
                    else:
                        t_1 = [0]
                else:
                    if list_first[0] == 0 and list_first[1] == 0:
                        t_1 = []
                    elif list_first[0] == 0 and list_first[1] == 1:
                        t_1 = [1]
                    elif list_first[0] == 1 and list_first[1] == 0:
                        t_1 = [0]
                    else:
                        t_3 = [0, 1]
                        return t_3
            else:
                t_1, t_3 = self.calculation_od_status(list_first)

            long.append(len(t_1) + len(t_3))
            # if len(t_3) > 0:
            for a in t_1:
                if isinstance(a, list):
                    for i in a:
                        times.append(time[i].hour)
                else:
                    times.append(time[a].hour)
            for b in t_3:
                for j in b:
                    times.append(time[j].hour)

            if len(t_3) == 1:
                for j in t_3:
                    # time_analysis[j[-1]] 是最后一个数的时间，time_analysis[j[0]]是第一个数的时间
                    time_intervals = (time[j[-1]] - time[j[0]]).seconds
                    df.loc[time[j[0]].hour, 'time_intervals'] = time_intervals

            else:
                time_intervals = 0
                m = 0
                for q, p in zip(long, t_3):
                    m = time[p[0]].hour
                    time_intervals = time_intervals + (time[p[-1]] - time[p[0]]).seconds
                if time_intervals != 0:
                    df.loc[m, 'time_intervals'] = time_intervals

        long = [i for i in long if i != 0]
        if len(times) > 0:
            for i, j in zip(set(times), long):
                df.loc[i, 'long'] = j
        else:
            for i in f:
                df.loc[i, 'long'] = 0
        for k, o in zip(f, taxi_num):
            df.loc[k, 'taxi_id'] = o
        return df

    def get_new_data(self, data, ps, num, parameter):
        """
            建立新的DataFrame, columns=['taxi_id','0',...,'23']
        :param data:
        :param ps:
        :param num:
        :param parameter:
        :return:
        """
        ps.loc[num, 'taxi_id'] = data.loc[0, 'taxi_id']
        for j in range(0, 24):
            ps.loc[num, '{0}'.format(j)] = data.loc[j, parameter]
        return ps

    def carry_passengers_main(self, data):
        """
            载客时长、次数算法的入口函数
        :param data:
        :return:
        """
        grouped = data.groupby('taxi_id')
        col = ['{}'.format(i) for i in range(0, 24)]
        col.insert(0, 'taxi_id')
        ps_time = pd.DataFrame([[0] * 25], columns=col)
        ps_long = pd.DataFrame([[0] * 25], columns=col)
        num = 0
        new_data_time = pd.DataFrame()
        new_data_long = pd.DataFrame()
        for index, df in grouped:
            status_list = []
            date_list = []
            t_1 = []
            t_2 = []
            grouped_status = df.groupby('hour')['status']
            grouped_datetime = df.groupby('hour')['date_time']

            for i, j in grouped_status:
                for m in j:
                    status_list.append(m)
                t_1.append(status_list)
                status_list = []

            for x, y in grouped_datetime:
                for n in y:
                    date_list.append(n)
                t_2.append(date_list)
                date_list = []

            df_1 = self.carry_passengers_static(zip(t_1, t_2), df)
            df_2 = self.carry_passengers_static(zip(t_1, t_2), df)
            if isinstance(df_1, list) is False:
                new_data_time = self.get_new_data(df_1, ps_time, num, 'time_intervals')
                new_data_long = self.get_new_data(df_2, ps_long, num, 'long')
            num = num + 1
        new_data_time = pd.DataFrame(new_data_time).reset_index(drop=True)
        new_data_long = pd.DataFrame(new_data_long).reset_index(drop=True)
        return new_data_time, new_data_long

    def process_data(self, c, day, num):
        """
            根据预处理规则处理数据
        :param c: 分块得到的数据
        :param day: 第几天的数据
        :param num: 第num个文件
        :return: 出租车总数量和载客出租车的数量
        """
        filename = "part-m-{}".format(num)
        df = pd.DataFrame()
        df = pd.concat([df, c], sort=False)
        taxi_id = [i for i in df.groupby(['taxi_id'])['taxi_id'].count().index]
        total_taxi = len(taxi_id)

        if int(day.split('_')[-1]) > 7:
            df.loc[:, 'date_time'] = df['date_time'].apply(lambda x: str(x)[:-6])
            df['date_time'] = pd.to_datetime(df['date_time'])

        print("{0} {1} 共有出租车：".format(day, filename), total_taxi, "辆")
        # 1、剔除字段为空或者内容为空的数据
        df.dropna(how='all', inplace=True)

        # 2、剔除'lng','lat'字段中重复的数据
        df.drop_duplicates(subset=['lng', 'lat'], keep='first', inplace=True)

        # 3、剔除行驶速度超过120迈和小于0迈的数据
        df = df[(df['speed'] < 120) & (df['speed'] > 0)]

        # 4、剔除行驶状态小于10的轨迹,即轨迹点数量小于10的taxi_id
        for i, status in df.groupby('taxi_id')['status']:
            if len(status) < 10:
                df = df[~(df['taxi_id'].isin([i]))]

        # 5、剔除空驶数据，如果一辆出租车全天的载客状态都为0，即视为空驶，给予剔除
        df.loc[:, 'hour'] = df.loc[:, 'date_time'].dt.hour
        new_data_time, new_data_long = self.carry_passengers_main(df)

        # 将所有的Nan值填充为0
        new_data_long = new_data_long.fillna(0)
        data_long = new_data_long.apply(lambda x: x.value_counts().get(0, 0), axis=1)  # 统计一行数据中0的个数
        # data_long = (new_data_long == 0).sum(axis=1)

        # 如果统计有24个0，即为全天空驶，则剔除该 taxi_id 的数据, 通过选取某值所在的全部行，然后进行取反，得到过滤后的数据
        empty_driver = []
        for index, values in enumerate(data_long):
            if values == 24:
                empty_driver.append(new_data_long.loc[index, 'taxi_id'])
                df = df[~(df['taxi_id'].isin([new_data_long.loc[index, 'taxi_id']]))]
                # df = df[~(df['taxi_id'].str.contains('{}'.format(new_data_long.loc[index,'taxi_id'])))]
        empty_taxi = len(empty_driver)
        print("空驶出租车的数量为: {} 辆".format(empty_taxi))

        # 6、对得到的data按照taxi_id分组，并对date_time进行排序
        dt = pd.DataFrame()
        df = df.groupby('taxi_id')
        for index, d in df:
            d = d.sort_values('date_time')
            dt = dt.append(d)
        dt.reset_index(drop=True, inplace=True)

        # 7、剔除上下车次数小于5次的数据
        id_list = self.delete_od_point(dt)
        track_num = 1
        for i in id_list:
            dt = dt[~(dt['taxi_id'].str.contains('{}'.format(i)))]

        od_data, match_data = self.od_extract_main(dt, track_num)
        match_data.to_csv(r'vv.csv', index=False)


        # # 8、对提取的出租车载客轨迹进行上下车等数据统计
        # zaike_info, zaike_duration, up_point, off_point = self.statis_duration_od(od_data)  # 返回载客时长、OD点次数数据集
        #
        # # 存储 zaike_track 表 (待匹配轨迹表）
        # os.chdir(os_path)
        # os.chdir(Const.PATH_TRACK)
        # if os.path.exists(day):
        #     match_data.to_csv(os.getcwd() + '\\' + day + '\\' + '{}.csv'.format(filename), index=False)
        # else:
        #     print("路径不存在.")
        #
        # # 存储 zaike_info 表  （载客信息表）
        # os.chdir(os_path)
        # os.chdir(Const.PATH_INFO)
        # if os.path.exists(day):
        #     zaike_info.to_csv(os.getcwd() + '\\' + day + '\\' + '{}.csv'.format(filename), index=False)
        # else:
        #     print("路径不存在.")
        #
        # # 存储 zaike_duration 表  （载客时间表）
        # os.chdir(os_path)
        # os.chdir(Const.PATH_DURATION)
        # if os.path.exists(day):
        #     zaike_duration.to_csv(os.getcwd() + '\\' + day + '\\' + '{}.csv'.format(filename), index=False)
        # else:
        #     print("路径不存在.")
        #
        # # 存储 up_point 表  （上客点表）
        # os.chdir(os_path)
        # os.chdir(Const.PATH_UP)
        # if os.path.exists(day):
        #     up_point.to_csv(os.getcwd() + '\\' + day + '\\' + '{}.csv'.format(filename), index=False)
        # else:
        #     print("路径不存在.")
        #
        # # 存储 off_point 表 （下客点表）
        # os.chdir(os_path)
        # os.chdir(Const.PATH_OFF)
        # if os.path.exists(day):
        #     off_point.to_csv(os.getcwd() + '\\' + day + '\\' + '{}.csv'.format(filename), index=False)
        # else:
        #     print("路径不存在.")
        #
        # # 存储 od_data 表 （包含上下车信息的OD表）
        # os.chdir(os_path)
        # os.chdir(Const.PATH_TRAJECTORY)
        # if os.path.exists(day):
        #     od_data.to_csv(os.getcwd() + '\\' + day + '\\' + '{}.csv'.format(filename), index=False)
        # else:
        #     print("路径不存在.")
        #
        # # 存储 processed_data 表（预处理完成表，未提取OD信息的表）
        # os.chdir(os_path)
        # os.chdir(Const.PATH_PROCESSED)
        # if os.path.exists(day):
        #     dt.to_csv(os.getcwd() + '\\' + day + '\\' + '{}.csv'.format(filename), index=False)
        # else:
        #     print("路径不存在.")
        #
        # # 存储 carrytime_data 表 （一辆出租车一天运营时间的起点和终点）
        # carraytime_data = self.extract_carray_time(dt)
        # os.chdir(os_path)
        # os.chdir(Const.PATH_CARRYTIME)
        # if os.path.exists(day):
        #     carraytime_data.to_csv(os.getcwd() + '\\' + day + '\\' + '{}.csv'.format(filename), index=False)
        # else:
        #     print("路径不存在.")
        #
        # # 存储 final_zaike_info 表 （一辆出租车一天的最终运营时间表）
        # final_data = self.compare_data(zaike_info, carraytime_data)
        # os.chdir(os_path)
        # os.chdir(Const.PATH_FINALTIME)
        # if os.path.exists(day):
        #     final_data.to_csv(os.getcwd() + '\\' + day + '\\' + '{}.csv'.format(filename), index=False)
        # else:
        #     print("路径不存在.")
        # os.chdir(os_path)
        # return total_taxi, empty_taxi

if __name__ == "__main__":
    ITERATION = Const.ITERATION
    tdp = TrajectoryDataProcess(ITERATION)
    tdp.run()