import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 读取CSV文件
df = pd.read_csv(r'C:\Users\24451\Documents\Tencent Files\2445163903\FileRecv\random_all_news.csv')

# 打印行数和列数
print("行数:", df.shape[0])
print("列数:", df.shape[1])

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 获取标签
labels = df.label

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# 初始化一个CountVectorizer，使用n-gram特征提取
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')

# 将训练集和测试集转换为n-gram向量
ngram_train = ngram_vectorizer.fit_transform(x_train)
ngram_test = ngram_vectorizer.transform(x_test)

# 初始化一个PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)

# 训练分类器
pac.fit(ngram_train, y_train)

# 预测测试集并计算准确率
y_pred = pac.predict(ngram_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')
# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)