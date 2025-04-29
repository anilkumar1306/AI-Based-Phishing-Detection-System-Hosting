from flask import Flask, request, jsonify, render_template
import re
import pickle
import tensorflow as tf, requests
import base64
import os
port = int(os.environ.get("PORT", 10000))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Load models and tokenizer
cnn_model = tf.keras.models.load_model('best_cnn.keras')
lstm_model = tf.keras.models.load_model('best_lstm.keras')
rnn_model = tf.keras.models.load_model('best_rnn.keras')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# VirusTotal API key
VIRUSTOTAL_API_KEY = 'ee6e9534c372447877b0c5cf1c3983fc54970250e3815403baf5db606aa53576'
max_length = 200  # Match your model's training sequence length
threshold = 0.5   # Classification threshold

def preprocess_text(text):
    """Clean email text before tokenization"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)          # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # Remove special chars/numbers
    return text.strip()

max_length = 200  # Match your model's training sequence length
threshold = 0.5   # Classification threshold
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']

    # Preprocess and predict
    processed_text = preprocess_text(email_text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')


    def get_prediction_label(model, padded_sequence, threshold):
      prediction = model.predict(padded_sequence, verbose=0)[0][0]
      return "Phishing" if prediction > threshold else "Legitimate"



    # Get model predictions
    cnn_pred = cnn_model.predict(padded_sequence, verbose=0)[0][0]
    lstm_pred = lstm_model.predict(padded_sequence, verbose=0)[0][0]
    rnn_pred = rnn_model.predict(padded_sequence, verbose=0)[0][0]

    # Calculate weighted ensemble
    weights = [0.33, 0.33, 0.34]
    ensemble_score = (weights[0] * cnn_pred + 
                     weights[1] * lstm_pred + 
                     weights[2] * rnn_pred)
    
    # Generate results
    
    results = {
        "CNN": get_prediction_label(cnn_model, padded_sequence, threshold),
        "LSTM": get_prediction_label(lstm_model, padded_sequence, threshold),
        "RNN": get_prediction_label(rnn_model, padded_sequence, threshold),
        "FinalVerdict": "Phishing" if ensemble_score > threshold else "Legitimate"}

    return jsonify(results)

# ... keep your existing about/team routes ...

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port,debug=True)