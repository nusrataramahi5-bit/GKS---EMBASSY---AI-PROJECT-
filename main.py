# GKS Embassy Track - Korea University AI Project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label','message'])
df['label'] = df.label.map({'ham':0, 'spam':1})
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
vec = TfidfVectorizer()
model = MultinomialNB()
model.fit(vec.fit_transform(X_train), y_train)
acc = accuracy_score(y_test, model.predict(vec.transform(X_test)))*100
print(f"Accuracy: {acc:.2f}%")
