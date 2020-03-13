# -*- coding: utf-8 -*-
# @Time    : 2019/7/12 15:42
# @Author  : xuzhifeng
# @File    : OriginalDataProcess.py
import pandas as pd
import os
import gc
import time
import multiprocessing as mp

os_path = os.getcwd()

class OriginalDataProcess(object):
    def __init__(self):
        pass

    def run(self):
        start_time = time.time()
        pool = mp.Pool(10)
        day_list = [_ for _ in range(8, 18)]
        for day in day_list:
            pool.apply_async(func = self.transform_txt_csv, args = (day, ))
        pool.close()
        pool.join()
        over_time = time.time()
        print("总共用时：", over_time - start_time)

    def transform_txt_csv(self, day):
        """
             将原始txt数据转换成csv格式
        :return:
        """
        full_files = []
        for root, dirc, filename in os.walk(r'J:/data_all/original_link_data/process_before/2015-05-{}.gz'.format(day)):
            for name in filename:
                full_files.append(os.path.join(root, name))
        data = []
        i = 1

        for index, file in enumerate(full_files):
            file_name = str(file.split('\\')[-3]).split('-')[-1].split('.')[0]
            path_name = "2015-5-{}".format(file_name)
            data.append(pd.read_csv(file, sep=',', header=None, parse_dates=[5], infer_datetime_format=True,
                                    names=['taxi_id', 'taxi_color', 'taxi_company', 'lng', 'lat',
                                           'date_time', 'c3', 'speed', 'direction', 'c6', 'c7', 'c8',
                                           ' status', 'license_color']))

            data = pd.concat(data, ignore_index=True)
            del data['taxi_color']
            del data['taxi_company']
            del data['c3']
            del data['c6']
            del data['c7']
            del data['c8']
            del data['license_color']

            os.chdir(r'J:\data_all\original_data\process_later')
            if os.path.exists(path_name) is False:
                os.mkdir(path_name)
            os.chdir(os.getcwd() + os.path.sep + path_name)
            pathname = "sz_{0}_{1}.csv".format(file_name, i)
            print("正在转换：sz_{0}_{1}".format(file_name, i))
            data.to_csv(pathname, index=False)
            data = []
            gc.collect()
            i += 1
        print("2015-5-{}转换结束，开始合并数据集".format(day))
        os.chdir(r'J:\data_all\original_data\process_later\2015-5-{}'.format(day))

        all_files = []
        for root, dirc, filename in os.walk(os.getcwd()):
            for f in filename:
                all_files.append(os.path.join(root, f))

        data = pd.read_csv(all_files[0])
        data.to_csv('J:/data_all/original_link_data/process_final/sz_5_%s.csv' % day, encoding="utf_8", index=False, header=1,
                    columns=['taxi_id', 'lng', 'lat', 'date_time', 'speed', 'direction', 'status'])

        for n in range(1, len(all_files)):
            dt = pd.read_csv(all_files[n])
            dt.to_csv('J:/data_all/original_link_data/process_final/sz_5_%s.csv' % day, encoding="utf_8", index=False, header=0,
                      mode='a+')
    os.chdir(os_path)

if __name__ == "__main__":
    odp = OriginalDataProcess()
    odp.run()