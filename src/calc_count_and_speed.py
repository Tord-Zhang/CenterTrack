import numpy as np 
import glob 
import os 


fps = 25
car_size = (4.8 + 1.8) / 2.   #一般小汽车的长宽
#假设帧率是25fps
def calc_count_and_speed(data):
    results = []
    max_track_id = 0
    #拿到整个视频中所有车辆的数量
    for d in range(len(data)-10, len(data)):
        for x in data[d]:
            if x["tracking_id"] > max_track_id:
                max_track_id = x["tracking_id"]
    
    center_point = np.ones((max_track_id, len(data), 2)) * -1
    
    count = 0
    size_sum = 0
    #计算视频中车辆的平均大小
    for key, frame_val in data.items():
        for i, obj_val in enumerate(frame_val):  
            size_sum += np.abs(obj_val["bbox"][2] - obj_val["bbox"][0])
            size_sum += np.abs(obj_val["bbox"][3] - obj_val["bbox"][1])
            count += 2
    car_pxl_size = size_sum / count 
    
    #整理每辆车的中心点坐标
    for key, frame_val in data.items():
        for i, obj_val in enumerate(frame_val):
            bbox = obj_val["bbox"]
            cpt = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
            t_id = obj_val["tracking_id"]
            center_point[t_id-1][key-1] = cpt 
    

    #计算每一帧，每一个车的方向
    direction = np.ones((max_track_id, len(data)-1)) * -1

    for i, cp in enumerate(center_point):
        g_x = cp[:, 0] - np.roll(cp[:, 0], 1, axis=0)
        g_x = g_x[1:]
        g_y = cp[:, 1] - np.roll(cp[:, 1], 1, axis=0)
        g_y = g_y[1:]
        cp_dirc = (g_x > g_y).astype(int)
        cp_dirc[cp[1:, 1]==-1] = -1
        direction[i] = cp_dirc

    #接下来计算车流量和速度
    #前10帧的所有数据都是0（数据不足以用来计算速度和流量）
    result = {
        "flow": [(0, 0)] * 10,
        "speed": [(0, 0)] * 10
    }

    for idx in range(10, len(data)-1):
        start_idx = max( idx - fps, 0)
        flow_x = direction[:, idx] == 1
        flow_y = direction[:, idx] == 0
        for i in range(start_idx, idx):
            tmp_x = direction[:, start_idx] == 1
            flow_x = np.bitwise_or(flow_x, tmp_x)
            tmp_y = direction[:, start_idx] == 0
            flow_y = np.bitwise_or(flow_y, tmp_y)    
        
        flow_x = np.sum(flow_x.astype(int)) / (idx - start_idx + 1) * fps
        flow_y = np.sum(flow_y.astype(int)) / (idx- start_idx + 1) * fps
        result["flow"].append((flow_x, flow_y))

        total_offset_x = 0
        total_offset_y = 0
        count_x = 0
        count_y = 0
        for i, x in enumerate(direction[:, idx]):
            offset_x = 0
            offset_y = 0
            if x  == 1 and direction[i, idx-1] != -1:
                offset_x = np.sqrt((center_point[i][idx][0] - center_point[i][idx-1][0])**2 + \
                                (center_point[i][idx][1] - center_point[i][idx-1][1])**2)
                count_x += 1
            else:
                offset_x = 0
            
            if x  == 0 and direction[i, idx] != -1:
                offset_y = np.sqrt((center_point[i][idx][0] - center_point[i][idx-1][0])**2 + \
                                (center_point[i][idx][1] - center_point[i][idx-1][1])**2)
                count_y += 1
            else:
                offset_y = 0
            
            total_offset_x += offset_x
            total_offset_y += offset_y
        

        if count_x == 0:
            avg_speed_x = 0
        else:
            avg_speed_x = float(total_offset_x) / car_pxl_size * car_size / count_x
        if count_y == 0:
            avg_speed_y = 0
        else:
            avg_speed_y = float(total_offset_y) / car_pxl_size * car_size / count_y

        result["speed"].append((avg_speed_x, avg_speed_y))
    
    new_result = {
        "flow": np.array(result["flow"]),
        "speed": np.array(result["speed"])
    }
    result["speed"] = np.array(result["speed"])

    for idx in range(10, len(data)-1):
        start_idx = max( idx - fps, 0)
        speed_x = np.sum(result["speed"][start_idx:idx+1][0]) / (idx - start_idx + 1)  * fps
        speed_y = np.sum(result["speed"][start_idx:idx+1][1]) / (idx - start_idx + 1) * fps
        
        new_result["speed"][idx]= np.array((speed_x, speed_y))
    
    return new_result


if __name__ == "__main__":
    data = np.load("results.npy", allow_pickle=True).item()
    results = calc_count_and_speed(data)
    for i in range(len(results["flow"])):
        print("frame {}, flow x: {}, flow y:{}, avg speed x: {:.2f} m/s, avg speed {:.2f} m/s".format(i, 
            int(results["flow"][i][0]), int(results["flow"][i][1]),results["speed"][i][0], results["speed"][i][1]))