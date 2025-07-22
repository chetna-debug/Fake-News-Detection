import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load and merge datasets
fake_df = pd.read_csv(r'C:\Users\cheth\Downloads\archive (2)\News _dataset\Fake.csv')
true_df = pd.read_csv(r'C:\Users\cheth\Downloads\archive (2)\News _dataset\True.csv')

fake_df['label'] = 1  # 1 = Fake
true_df['label'] = 0  # 0 = Real

df = pd.concat([fake_df, true_df], ignore_index=True)

# 2. Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    return text.strip()

df['text'] = df['text'].apply(clean_text)

# 3. Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 4. Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Save the model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and vectorizer saved successfully!")
