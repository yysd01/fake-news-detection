import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Read CSV file
csv_file_path = 'random_all_news.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# View the first few rows of the data
print(df.head())

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase and split into words
    words = text.lower().split()

    # Remove punctuation
    words = [word.strip(string.punctuation) for word in words]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Apply stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Apply text preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Train Word2Vec model
model = Word2Vec(sentences=df['processed_text'].tolist(), vector_size=100, window=5, min_count=1, workers=4)


# Document vector representation
def document_vector(word2vec_model, doc):
    return np.mean([word2vec_model.wv[word] for word in doc if word in word2vec_model.wv] or [np.zeros(word2vec_model.vector_size)], axis=0)


# Create feature vectors
X = np.array([document_vector(model, doc) for doc in df['processed_text']])
y = df['label']  # Assuming there is a 'label' column in the CSV containing class labels

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Support Vector Machine model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict
y_pred = svm_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
