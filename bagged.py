import os
import glob
import pandas as pd
# 读取CSV文件
df = pd.read_csv(r'G:\qq\news\random_all_news.csv')
# 打印行数和列数
print("行数:", df.shape[0])
print("列数:", df.shape[1])
#
# print(df.loc[0, 'text'])
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 读取数据
# df=pd.read_csv('news.csv')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# 获取标签
labels=df.label
# print(labels)
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# 初始化一个tfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# 将训练集和测试集转换为tfidf向量
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

# print("tfidf_train=:\n",tfidf_train)
# 初始化一个PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# 预测测试集并计算准确率
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#输入文本
text = input("请输入你想要预测的文本：")

# 将文本转换为tfidf向量
tfidf_text = tfidf_vectorizer.transform([text])

# 使用模型进行预测
prediction = pac.predict(tfidf_text)
print(prediction[0])
# 打印预测结果
if prediction[0] == 1:
    print("这是假新闻")
elif prediction[0] == 0:
    print("这是真新闻")