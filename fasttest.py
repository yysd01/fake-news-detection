import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split
# 读取CSV文件
df = pd.read_csv(r'G:\qq\news\random_all_news.csv')
# 读取数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# 获取标签并转换格式
df['label'] = '__label__' + df['label'].astype(str)
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=42)
# 将训练数据和测试数据保存为fastText需要的格式
train_data = pd.concat([y_train, x_train], axis=1)
test_data = pd.concat([y_test, x_test], axis=1)
train_data.to_csv(r'G:\qq\news\train.txt', index=False, sep=' ', header=None)
test_data.to_csv(r'G:\qq\news\test.txt', index=False, sep=' ', header=None)
# 使用fastText训练模型
model = fasttext.train_supervised(input=r'G:\qq\news\train.txt', lr=0.25, epoch=25, wordNgrams=1)
# 预测测试集并计算准确率
result = model.test(r'G:\qq\news\test.txt')
print(result)
print(f'Accuracy: {round(result[1]*100,2)}%')
# 输入文本
while(1):
    text = input("请输入你想要预测的文本：")
# 使用模型进行预测
    labels, probabilities = model.predict(text)
# 打印预测结果
    if labels[0] == '__label__1':
        print("这是假新闻")
    elif labels[0] == '__label__0':
        print("这是真新闻")
CNN: Accuracy: 98.200%（卷积核大小为3，词嵌入维度128）
# 用CNN实现文本分类
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
# 读取CSV文件
df = pd.read_csv(r'G:\qq\news\random_all_news.csv')
# 读取数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# 获取标签并转换格式
df['label'] = df['label'].astype(int)
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=42)
# 文本预处理
max_words = 10000
maxlen = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(sequences, maxlen=maxlen)
sequences = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(sequences, maxlen=maxlen)
# 构建CNN模型
model = Sequential()
model.add(Embedding(max_words, 128, input_length=maxlen))
model.add(Conv1D(32, 3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
# 编译模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# 训练模型
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy*100:.3f}%')
 
Textcnn：
# 使用TextCNN实现文本分类
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, concatenate
from keras.optimizers import Adam
# 读取CSV文件
df = pd.read_csv(r'G:\qq\news\random_all_news.csv')
# 读取数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# 获取标签并转换格式
df['label'] = df['label'].astype(int)
labels = df.label
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=42)
# 文本预处理
max_words = 10000
maxlen = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(sequences, maxlen=maxlen)
sequences = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(sequences, maxlen=maxlen)
# 构建TextCNN模型
embedding_dim = 128
filter_sizes = [3, 4, 5]
num_filters = 128
drop = 0.5
inputs = Input(shape=(maxlen,))
embedding = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=maxlen)(inputs)
conv_0 = Conv1D(num_filters, filter_sizes[0], activation='relu')(embedding)
conv_1 = Conv1D(num_filters, filter_sizes[1], activation='relu')(embedding)
conv_2 = Conv1D(num_filters, filter_sizes[2], activation='relu')(embedding)
maxpool_0 = GlobalMaxPooling1D()(conv_0)
maxpool_1 = GlobalMaxPooling1D()(conv_1)
maxpool_2 = GlobalMaxPooling1D()(conv_2)
concatenated_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
dropout = Dropout(drop)(concatenated_tensor)
output = Dense(1, activation='sigmoid')(dropout)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
# 训练模型
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy*100:.3f}%')