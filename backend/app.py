# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy
import sklearn
import flask_cors
import gunicorn

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

# New route to verify installed package versions
@app.route('/api/versions', methods=['GET'])
def versions():
    return jsonify({
        'numpy': numpy.__version__,
        'scikit-learn': sklearn.__version__,
        'Flask': Flask.__version__,
        'flask-cors': flask_cors.__version__,
        'gunicorn': gunicorn.__version__
    })    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
