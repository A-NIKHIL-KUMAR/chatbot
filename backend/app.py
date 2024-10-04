# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend to communicate with backend

# Load your pre-trained model and vectorizer 
model = pickle.load(open('backend/model_response.pkl', 'rb'))
vectorizer = pickle.load(open('backend/vectorizer.pkl', 'rb'))

@app.route('/api/chatbot', methods=['POST'])
def chatbot_response():
    data = request.get_json()
    user_input = data.get('message')
    if not user_input:
        return jsonify({'response': 'Please provide a message.'}), 400

    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)
    return jsonify({'response': prediction[0]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
