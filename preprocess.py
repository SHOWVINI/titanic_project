# preprocess.py

import pandas as pd

# 데이터 로드
df = pd.read_csv('data/train.csv')

# 결측치 처리 예시
# 수치형 열만 선택하여 결측치를 평균으로 채움
num_cols = df.select_dtypes(include=['number']).columns  # 수치형 열 선택
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# 범주형 열의 결측치는 최빈값으로 채움
cat_cols = df.select_dtypes(include=['object']).columns  # 범주형 열 선택
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 모든 열을 원-핫 인코딩
df = pd.get_dummies(df, drop_first=True)

# 전처리된 데이터 저장
df.to_csv('data/train_preprocessed.csv', index=False)

