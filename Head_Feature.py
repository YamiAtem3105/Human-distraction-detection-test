import numpy as np
import pandas as pd

def extract_head_pose_features(data):
    """
    Trích xuất và chuẩn hóa các đặc trưng về tư thế đầu từ dữ liệu đầu vào.
    """
    # Kiểm tra xem các cột cần thiết có tồn tại không
    required_columns = ['pose_Rx', 'pose_Ry', 'pose_Rz']
    if not set(required_columns).issubset(data.columns):
        raise ValueError("Dữ liệu không chứa đủ các cột cần thiết.")

    # 1. Chuẩn hóa dữ liệu (Z-score Normalization)
    def z_score_normalization(values):
        mean_value = np.mean(values)
        std_value = np.std(values)
        normalized_values = (values - mean_value) / std_value
        return normalized_values

    normalized_pose_Rx = z_score_normalization(data['pose_Rx'])
    normalized_pose_Ry = z_score_normalization(data['pose_Ry'])
    normalized_pose_Rz = z_score_normalization(data['pose_Rz'])

    # Tạo DataFrame kết quả
    result = pd.DataFrame({
        'normalized_pose_Rx': normalized_pose_Rx,
        'normalized_pose_Ry': normalized_pose_Ry,
        'normalized_pose_Rz': normalized_pose_Rz
    })

    return result

# Ví dụ sử dụng
if __name__ == "__main__":

    data = pd.DataFrame({
        'pose_Rx': np.random.uniform(-30, 30, 300),  # Góc quay ngang
        'pose_Ry': np.random.uniform(-20, 20, 300),  # Góc quay dọc
        'pose_Rz': np.random.uniform(-15, 15, 300)   # Góc nghiêng
    })

    # Trích xuất đặc trưng tư thế đầu
    head_pose_features = extract_head_pose_features(data)
    print(head_pose_features.head())