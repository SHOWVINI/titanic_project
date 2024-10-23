# preprocess.py

import pandas as pd


df = pd.read_csv('data/train.csv')


# 수치형 열만 선택하여 결측치를 평균으로 
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# 범주형 열의 결측치는 최빈값으로
cat_cols = df.select_dtypes(include=['object']).columns 

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 모든 열 원-핫 인코딩
df = pd.get_dummies(df, drop_first=True)

# 전처리 데이터 저장
df.to_csv('data/train_preprocessed.csv', index=False)

