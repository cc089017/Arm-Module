# preprocessor.py
import pandas as pd

def load_and_preprocess(csv_path: str):
    """
    CSV 하나를 읽어서 pronator drift/drop feature와 이진 레이블(0/1)을
    반환합니다.
    """
    df = pd.read_csv(csv_path)

    # both_abnormal 제외
    df = df[df['final_diagnosis'] != 'both_abnormal']
    df = df[df['final_diagnosis'].notna()]

    # 이진 라벨 생성
    df['label'] = df['final_diagnosis'].apply(lambda x: 0 if x == 'normal' else 1)

    # feature 컬럼
    feature_cols = [
        'left_start_slope', 'left_end_slope', 'left_slope_diff',
        'right_start_slope', 'right_end_slope', 'right_slope_diff',
        'left_y0', 'left_y1', 'left_y2', 'left_y3', 'left_y4',
        'right_y0', 'right_y1', 'right_y2', 'right_y3', 'right_y4',
    ]

    X = df[feature_cols].fillna(0)
    y = df['label']
    return X, y
