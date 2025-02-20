import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def extract_mouth_features(data):
    
    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = ['mouth_lmk_x_61', 'mouth_lmk_y_61', 'mouth_lmk_x_67', 'mouth_lmk_y_67']
    if not set(required_columns).issubset(data.columns):
        raise ValueError("Dữ liệu không chứa đủ các cột cần thiết.")

    # 1. Tính mức độ mở miệng (Mouth Open)
    upper_lip_x = data['mouth_lmk_x_61'].values
    upper_lip_y = data['mouth_lmk_y_61'].values
    lower_lip_x = data['mouth_lmk_x_67'].values
    lower_lip_y = data['mouth_lmk_y_67'].values

    mouth_open = np.sqrt((upper_lip_x - lower_lip_x)**2 + (upper_lip_y - lower_lip_y)**2)

    # 2. Tính diện tích miệng (Mouth Area)
    mouth_landmarks = data[['mouth_lmk_x_61', 'mouth_lmk_y_61', 'mouth_lmk_x_62', 'mouth_lmk_y_62',
                            'mouth_lmk_x_63', 'mouth_lmk_y_63', 'mouth_lmk_x_64', 'mouth_lmk_y_64',
                            'mouth_lmk_x_65', 'mouth_lmk_y_65', 'mouth_lmk_x_66', 'mouth_lmk_y_66',
                            'mouth_lmk_x_67', 'mouth_lmk_y_67']].values
    mouth_area = np.array([calculate_polygon_area(points.reshape(-1, 2)) for points in mouth_landmarks])

    # 3. Tần suất ngáp (Yawning Rate)
    yawning_rate = (mouth_open > 0.8).sum() / len(data)  # Giả sử ngáp khi mouth_open > 0.8

    # 4. Dao động mở miệng (Mouth Variance)
    mouth_variance = np.var(mouth_open)

    # Chuẩn hóa dữ liệu (Min-Max Normalization)
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(np.array([
        mouth_open,
        mouth_area,
        [yawning_rate] * len(mouth_open),
        [mouth_variance] * len(mouth_open)
    ]).T)

    # Tạo DataFrame kết quả
    result = pd.DataFrame(normalized_features, columns=[
        'normalized_mouth_open',
        'normalized_mouth_area',
        'normalized_yawning_rate',
        'normalized_mouth_variance'
    ])

    return result

def calculate_polygon_area(points):
    
    # Tính diện tích đa giác từ các điểm.
    n = len(points)
    area = 0.5 * abs(sum(points[i][0] * points[(i + 1) % n][1] - points[i][1] * points[(i + 1) % n][0] for i in range(n)))
    return area

# Ví dụ sử dụng
if __name__ == "__main__":

    data = pd.DataFrame({
        'mouth_lmk_x_61': np.random.uniform(50, 100, 300),  # Tọa độ X của môi trên
        'mouth_lmk_y_61': np.random.uniform(50, 100, 300),  # Tọa độ Y của môi trên
        'mouth_lmk_x_67': np.random.uniform(60, 110, 300),  # Tọa độ X của môi dưới
        'mouth_lmk_y_67': np.random.uniform(60, 110, 300)   # Tọa độ Y của môi dưới
    })

    # Trích xuất đặc trưng mức độ mở miệng
    mouth_features = extract_mouth_features(data)
    print(mouth_features.head())