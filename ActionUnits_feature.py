import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def extract_action_units(data):
    
    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU26_r']
    if not set(required_columns).issubset(data.columns):
        raise ValueError("Dữ liệu không chứa đủ các cột cần thiết.")

    # Chuẩn hóa dữ liệu (Min-Max Normalization)
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(data[required_columns])

    # Tạo DataFrame kết quả
    result = pd.DataFrame(normalized_features, columns=required_columns)

    return result

# Ví dụ sử dụng
if __name__ == "__main__":
   
    data = pd.DataFrame({
        'AU01_r': np.random.uniform(0, 1, 300),  # Inner Brow Raiser
        'AU02_r': np.random.uniform(0, 1, 300),  # Outer Brow Raiser
        'AU04_r': np.random.uniform(0, 1, 300),  # Brow Lowerer
        'AU05_r': np.random.uniform(0, 1, 300),  # Upper Lid Raiser
        'AU06_r': np.random.uniform(0, 1, 300),  # Cheek Raiser
        'AU07_r': np.random.uniform(0, 1, 300),  # Lid Tightener
        'AU12_r': np.random.uniform(0, 1, 300),  # Lip Corner Puller
        'AU26_r': np.random.uniform(0, 1, 300)   # Jaw Drop
    })

    # Trích xuất đặc trưng biểu cảm khuôn mặt
    au_features = extract_action_units(data)
    print(au_features.head())