from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and vectorizer
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'message' not in data:
            return jsonify({'error': 'Message key is missing in the request'}), 400

        message = data['message']

        # Convert the message to a count vector
        count_vector = vectorizer.transform([message])

        # Make a prediction
        prediction = model.predict(count_vector)

        # Return the result as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)