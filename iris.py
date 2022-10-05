# 基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# データセットの読み込み
iris = load_iris()
x = iris.data[:, [0, 2]]
y = iris.target
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# 目標値
df['target'] = iris.target
# 目標値から切り分け
df.loc[df['target'] == 0, 'target'] = 'setosa'
df.loc[df['target'] == 1, 'target'] = 'versicolor'
df.loc[df['target'] == 2, 'target'] = 'virginica'
# モデルの構築
clf = LogisticRegression()
clf.fit(x, y)

## 入力画面
st.sidebar.header('Input Features')
sepalValue = st.sidebar.slider('sepal Length(cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('petal length(cm)', min_value=0.0, max_value=10.0, step=0.1)

## メインパネル
st.title('Iris Classifier')
st.write('## Input Value')

## インプットデータ（１桁のデータフレーム）
value_df = pd.DataFrame([],columns=['data','sepal length (cm)','petal length (cm)'])
record = pd.Series(['data',sepalValue, petalValue], index=value_df.columns)
value_df = value_df.append(record, ignore_index=True)
value_df.set_index('data',inplace=True)

## 入力値
st.write(value_df)

## 予測値のデータフレーム
pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs, columns=['setosa', 'versicolor', 'virginica'], index=['probability'])

st.write('## Prediction')
st.write(pred_df)

## 予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write(f'このアイリスはきっと{str(name[0])}です。')

