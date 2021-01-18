import json
import requests
import numpy as np
import time
import pulp
import itertools
import folium
import copy
import pyproj
from joblib import Parallel, delayed
import datetime





def solve_DPP(spot_list, satisfy, t_array, spot_num, stay_time, limit,start_Day,stay_Days,start_hotel_hour_minutes):




    t_3 = time.time()

    SID = "Nqc7W4Y6YrLG"
    sleep = 0.1

    day = stay_Days # 日数
    one_day_limit = limit # 一日の所要時間上限

    day_1 = day + 1

    # モデルの構築
    prob = pulp.LpProblem("CVRP", pulp.LpMaximize)
    x = [[[pulp.LpVariable("x_%s_%s_%s"%(i,j,k), cat="Binary") if i != j else None for k in range(day_1)] for j in range(spot_num)] for i in range(spot_num)]

    #各観光スポットを訪れたかと，その満足度の掛け合わせ．回らない観光スポットを回る日は除外
    prob += pulp.lpSum(x[i][j][k] * satisfy[j] - 0.001 * x[i][j][k] * (t_array[i][j] - stay_time[j]) for i in range(spot_num) for j in range(spot_num) if i != j for k in range(day))

    #制約
    #(2)式各顧客の場所に訪れるのは１日のみ１度である
    for j in range(1, spot_num):
            prob += pulp.lpSum(x[i][j][k] if i != j else 0 for i in range(spot_num) for k in range(day_1)) == 1

    #(3)式, ホテルから出発して，ホテルに戻ってくる
    for k in range(day):
        # ホテルを出発した運搬車が必ず 1つの観光地から訪問を開始することを保証する制約条件
        prob += pulp.lpSum(x[0][j][k] for j in range(1,spot_num)) == 1
        # 必ず 1 つの観光地からホテルへ到着すること保証する制約条件
        prob += pulp.lpSum(x[i][0][k] for i in range(1,spot_num)) == 1

    #(4)式, ある顧客の所に来る車両数と出る車両数が同じ
    for k in range(day_1):
        for j in range(spot_num):
            prob += pulp.lpSum(x[i][j][k] if i != j else 0 for i in range(spot_num)) -  pulp.lpSum(x[j][i][k] for i in range(spot_num)) == 0

    #(5)式, 一日の移動上限を明記
    for k in range(day):
        prob += pulp.lpSum(x[i][j][k] * t_array[i][j] if i != j else 0 for i in range(spot_num) for j in range(spot_num)) <= one_day_limit

    #(7)式, 部分巡回路除去制約()
        #itertools.combinations（組み合わせ）i個選ぶ組み合わせ
    subtours = []
    for i in range(2,spot_num):
        subtours += itertools.combinations(range(1,spot_num), i)

    for s in subtours:
        prob += pulp.lpSum(x[i][j][k] if i !=j else 0 for i, j in itertools.permutations(s,2) for k in range(day)) <= len(s) - 1
    print("start solve...")
    # 実行
    prob.solve()


    t_4 = time.time()
    print("prob_solve ",t_4 - t_3)



    #変えました．測地系変更用
    def grs_to_bessel(jx,jy):
        wy = jy - jy * 0.00010695 + jx * 0.000017464 + 0.0046017
        wx = jx - jy * 0.000046038 - jx * 0.000083043 + 0.010040
        return wx ,wy

    def conv_bbox(bbox):
        return_bbox = [0,0,0,0]
        for i in range(2):
            return_bbox[i * 2 + 0],return_bbox[i * 2 + 1] = grs_to_bessel(bbox[i * 2 + 0],bbox[i * 2 + 1])
        return return_bbox

    def conv_coordinates(coordinates):
        return_coord = np.zeros_like(coordinates)
        for i in range(len(coordinates)):
            return_coord[i][0],return_coord[i][1] = grs_to_bessel(coordinates[i][0],coordinates[i][1])
        return_coord = return_coord.tolist()
        return return_coord
    #得られたjsonデータの緯度経度の測地系を変更
    def conv_j_func(j_data):
        return_j_data = copy.deepcopy(j_data)
        return_j_data['bbox'] = conv_bbox(j_data['bbox'])
        for i in range(len(j_data['features'])):
            return_j_data['features'][i]['bbox'] = conv_bbox(j_data['features'][i]['bbox'])
            return_j_data['features'][i]['geometry']['coordinates'] = conv_coordinates(j_data['features'][i]['geometry']['coordinates'])

        return return_j_data



    # 経路の同線取得のapiを叩く
    def get_shape(i, j,start_time,day_index):
        # 出発地点
        start = '{"lat":' + spot_list[i][0] + ',"lon":' + spot_list[i][1] + ',"name":"' + spot_list[i][2] + '"}'
        # 到着地点
        goal = '{"lat":' + spot_list[j][0] + ',"lon":' + spot_list[j][1] + ',"name":"' + spot_list[j][2] + '"}'

        #apiをたたくうえで時や分を2桁に統一
        len_hour = 1 if start_time[0] < 10 else 0
        len_min = 1 if start_time[1] < 10 else 0
        url = "https://api-challenge.navitime.biz/v1s/" + SID + "/route/shape?start=" + start + "&goal=" + goal + "&add=transport_shape&start-time="  + day_index +  "T" + "0" * len_hour + str(start_time[0]) + ":" + "0" * len_min+ str(start_time[1])

        r = requests.get(url) # apiを叩いてjsonデータを取得
        j_data = json.loads(r.text) # データ形式をデコード
        convert_json_data = conv_j_func(j_data) #jsonファイルを世界測地系に
        time.sleep(sleep)
        return convert_json_data

    #訪れる準のindexを取得
    def convert_cirtcuit(pass_list):
        circuit_route = []
        for k in range(day):
            oneday_circuit = [0]

            while len(pass_list[k]) != 0:
                j = 0
                while pass_list[k][j][0] != oneday_circuit[-1]:
                    j += 1
                oneday_circuit.append(pass_list[k][j][1])
                del pass_list[k][j]
            circuit_route.append(oneday_circuit)
        return circuit_route

    #改めて到着時間を検索．
    def get_time_api(i,j,start_time,day_index):
        # 出発地点
        print("b",start_time)
        start = '{"lat":' + spot_list[i][0] + ',"lon":' + spot_list[i][1] + ',"name":"' + spot_list[i][2] + '"}'
        # 到着地点
        goal = '{"lat":' + spot_list[j][0] + ',"lon":' + spot_list[j][1] + ',"name":"' + spot_list[j][2] + '"}'
        len_hour = 1 if start_time[0] < 10 else 0
        len_min = 1 if start_time[1] < 10 else 0


        # apiを叩く用のurl
        url = "https://api-challenge.navitime.biz/v1s/" + SID + "/route?start=" + start + "&goal=" + goal + "&start-time=" + day_index +  "T" + "0" * len_hour + str(start_time[0]) + ":" + "0" * len_min+ str(start_time[1])
        print(url)
        r = requests.get(url) # apiを叩いてjsonデータを取得
        j_data = json.loads(r.text) # データ形式をデコード
        print(j_data['items'][0]['summary']['move']['to_time'])
        del_ ,time_data = j_data['items'][0]['summary']['move']['to_time'].split("T")
        hour_ , minutes_ , del_ ,del_ = time_data.split(":")
        arrive_time = [int(hour_),int(minutes_)]
        return j_data['items'][0]['summary'],arrive_time



    def get_time_and_route(oneday_circuit_route,start_hotel_hour_minutes,day_str):
        start_hour = start_hotel_hour_minutes[0]
        start_minutes = start_hotel_hour_minutes[1]
        start_spot_times = []
        j_son_time_output_list = []
        for j in range(len(oneday_circuit_route) - 1):
            j_son_data , arrive_spot_time = get_time_api(oneday_circuit_route[j],oneday_circuit_route[j+1],[start_hour,start_minutes],day_str)
            start_spot_times.append([int(oneday_circuit_route[j]),int(oneday_circuit_route[j+1]),int(start_hour),int(start_minutes)])
            j_son_time_output_list.append(j_son_data)
            temp_arrive_time_list = list(j_son_data['move']['to_time'].split(":"))
            arrive_hour = int(temp_arrive_time_list[0][-2:])
            arrive_minutes = int(temp_arrive_time_list[1])
            start_hour = arrive_hour + ((arrive_minutes + stay_time[oneday_circuit_route[j+1]]) // 60)
            start_minutes = (arrive_minutes + stay_time[oneday_circuit_route[j+1]]) % 60
        print(start_spot_times)
        return j_son_time_output_list,start_spot_times




    pass_list = []
    lost_spot = [(x+1) for x in range(spot_num - 1)]
    for k in range(day):
        oneday_pass = []
        for i in range(spot_num):
            for j in range(spot_num):
                if i != j:
                    if pulp.value(x[i][j][k]) >= 1.e-4:
                        temp_1,temp_2,temp_3,temp_4 = str(x[i][j][k]).split("_")
                        oneday_pass.append([int(temp_2),int(temp_3)])
                        if j in lost_spot:
                            lost_spot.remove(j)
        pass_list.append(oneday_pass)
    circuit_route = convert_cirtcuit(pass_list)
    print("lost_spot",lost_spot)
    print(circuit_route)

    #ルートの来訪時間など
    j_route_time = []
    start_spot_times_array = []
    start_day_list = list(start_Day.split("-"))
    day_index = datetime.datetime(int(start_day_list[0]),int(start_day_list[1]),int(start_day_list[2]))
    for k in range(day):
        oneday_j_route_time , oneday_start_spot_times = (get_time_and_route(circuit_route[k],start_hotel_hour_minutes,day_index.strftime('%Y-%m-%d')))
        j_route_time.append(oneday_j_route_time)
        start_spot_times_array.append(oneday_start_spot_times)
        day_index += datetime.timedelta(days = 1)

    #ルート形状
    j_route_shape = []
    day_index = datetime.datetime(int(start_day_list[0]),int(start_day_list[1]),int(start_day_list[2]))

    for k in range(day):
        print("test",start_spot_times_array[k])
        j_route_shape.append(list((Parallel(n_jobs=-1)([delayed(get_shape)(c_[0],c_[1],[c_[2],c_[3]],day_index.strftime('%Y-%m-%d')) for c_ in start_spot_times_array[k]]))))


    t_5 = time.time()
    print("return j_data ",t_5 - t_4)
    lost_spot_name = []
    for i in lost_spot:
        lost_spot_name.append(spot_list[i][2])
    print(lost_spot_name)

    return j_route_time, j_route_shape ,lost_spot_name
