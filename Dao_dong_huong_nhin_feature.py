import numpy as np
import pandas as pd
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler

def extract_gaze_variance_features(data):
    
    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = ['gaze_angle_x', 'gaze_angle_y', 'blink_rate', 'pose_Rx']
    if not set(required_columns).issubset(data.columns):
        raise ValueError("Dữ liệu không chứa đủ các cột cần thiết.")

    # 1. Dao động góc nhìn ngang và dọc
    gaze_variance_x = np.var(data['gaze_angle_x'])
    gaze_variance_y = np.var(data['gaze_angle_y'])

    # 2. Chuyển động đầu (DCT trên Yaw Motion)
    yaw_motion_signal = data['pose_Rx'].values
    dct_yaw_motion = np.abs(dct(yaw_motion_signal))

    # Chuẩn hóa dữ liệu (Min-Max Normalization)
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(np.array([
        [gaze_variance_x],
        [gaze_variance_y],
        dct_yaw_motion
    ]).T)

    # Tạo DataFrame kết quả
    result = pd.DataFrame(normalized_features, columns=[
        'normalized_gaze_variance_x',
        'normalized_gaze_variance_y',
        'dct_yaw_motion'
    ])

    return result

# Ví dụ sử dụng
if __name__ == "__main__":
 
    data = pd.DataFrame({
        'gaze_angle_x': np.random.uniform(-30, 30, 300),  # Góc nhìn ngang
        'gaze_angle_y': np.random.uniform(-20, 20, 300),  # Góc nhìn dọc
        'pose_Rx': np.random.uniform(-10, 10, 300)        # Góc quay đầu (Yaw)
    })

    # Trích xuất đặc trưng dao động hướng nhìn
    gaze_variance_features = extract_gaze_variance_features(data)
    print(gaze_variance_features.head())