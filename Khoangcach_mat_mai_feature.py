import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_eye_nose_distance(data):
    
    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = ['eye_lmk_x_27', 'eye_lmk_y_27', 'nose_tip_x', 'nose_tip_y']
    if not set(required_columns).issubset(data.columns):
        raise ValueError("Dữ liệu không chứa đủ các cột cần thiết.")

    # 1. Tính khoảng cách giữa mắt và mũi
    # Giả sử eye_lmk_x_27, eye_lmk_y_27 là tọa độ mắt trái,
    # và nose_tip_x, nose_tip_y là tọa độ đỉnh mũi.
    eye_x = data['eye_lmk_x_27'].values
    eye_y = data['eye_lmk_y_27'].values
    nose_x = data['nose_tip_x'].values
    nose_y = data['nose_tip_y'].values

    # Tính khoảng cách Euclidean giữa mắt và mũi
    dist_eye_nose = np.sqrt((eye_x - nose_x)**2 + (eye_y - nose_y)**2)

    # 2. Chuẩn hóa dữ liệu (Z-score Normalization)
    scaler = StandardScaler()
    normalized_dist_eye_nose = scaler.fit_transform(dist_eye_nose.reshape(-1, 1)).flatten()

    # Tạo DataFrame kết quả
    result = pd.DataFrame({
        'dist_eye_nose': dist_eye_nose,
        'normalized_dist_eye_nose': normalized_dist_eye_nose
    })

    return result

# Ví dụ sử dụng
if __name__ == "__main__":
    data = pd.DataFrame({
        'eye_lmk_x_27': np.random.uniform(50, 100, 100),  # Tọa độ X của mắt trái
        'eye_lmk_y_27': np.random.uniform(50, 100, 100),  # Tọa độ Y của mắt trái
        'nose_tip_x': np.random.uniform(60, 110, 100),    # Tọa độ X của đỉnh mũi
        'nose_tip_y': np.random.uniform(60, 110, 100)     # Tọa độ Y của đỉnh mũi
    })

    # Trích xuất đặc trưng khoảng cách mắt - mũi
    eye_nose_features = extract_eye_nose_distance(data)
    print(eye_nose_features.head())