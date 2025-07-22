from flask import Flask, request, render_template
import pickle
import re

app = Flask(__name__)

# Load the saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text cleaning (same as before)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news']
    cleaned = clean_text(input_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = '🔴 Fake News' if prediction == 1 else '🟢 Real News'
    return render_template('index.html', result=result, original=input_text)

if __name__ == '__main__':
    app.run(debug=True)
