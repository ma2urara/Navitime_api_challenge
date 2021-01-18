import json
import requests
import numpy as np
import time
import copy
from joblib import Parallel, delayed

#距離(時間)行列の取得
def make_time_mat(spot_list, stay_time):

    t_start = time.time()

    SID = "Nqc7W4Y6YrLG"

    #一度にとれるペアの最大数8
    max_num = 8
    #時間を置く秒数
    sleep = 0.1
    #巡回地点の数
    spot_num = len(spot_list)







    # spot_listのindex
    #spot_index = [n for n in range(spot_num)]

    # 各地点の名前
    #spot_name = [n[2] for n in spot_list]

    # 所要時間行列

    # apiを叩く組み合わせを追加
    sg_list = []
    for s in range(spot_num-1):
        g_list = []
        for g in range(s+1, spot_num):
            # goalの数が8個の場合
            if (g-s)%max_num == 0 and g != spot_num-1:
                g_list.append(g) # goalに要素を追加
                sg_list.append([s, g_list]) # apiを叩く組み合わせを追加
                g_list = []
            else:
                g_list.append(g)
        sg_list.append([s, g_list])
    # urlの作成
    def call_api(s_index, g_index):
        t_array = np.zeros((spot_num, spot_num), dtype="int")
        # 出発地点
        start = '{"lat":' + spot_list[s_index][0] + ',"lon":' + spot_list[s_index][1] + ',"name":"' + spot_list[s_index][2] + '"}'
        # 到着地点
        goal = '['
        for g in g_index:
            goal += '{"lat":' + spot_list[g][0] + ',"lon":' + spot_list[g][1] + ',"name":"' + spot_list[g][2] + '"},'
        goal = goal[:-1] + ']'

        # 作成されたurl
        url = 'https://api-challenge.navitime.biz/v1s/' + SID + '/route?start=' + start + '&goal=' + goal

        #apiを叩いて所要時間を取得
        r = requests.get(url) # apiを叩いてjsonデータを取得

        j_data = json.loads(r.text) # データ形式をデコード

        # 所要時間を更新
        if len(g_index) == 1:
            distance = j_data['items'][0]['summary']['move']['time']
            t_array[s_index][g_index[0]] = distance
            t_array[g_index[0]][s_index] = distance
        else:
            for g, g_num in zip(j_data['items'], range(len(j_data['items']))):
                distance = g['summary']['move']['time']
                t_array[s_index][g_index[g_num]] = distance
                t_array[g_index[g_num]][s_index] = distance
        return t_array
        time.sleep(sleep)

    #実行部分


    output_t = np.sum(Parallel(n_jobs=-1)([delayed(call_api)(sg[0],sg[1]) for sg in sg_list]),axis=0)
    print(output_t)
    for i in range(len(spot_list)):
        output_t[:,i] += stay_time[i]
    t_array = output_t
    print(t_array)

    t_end = time.time()
    print("preprocessing ",t_end - t_start)

    return t_array
