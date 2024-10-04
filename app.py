from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('chat2.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json.get('message')
    input_vec = vectorizer.transform([user_input])  # Vectorize user input
    prediction = model.predict(input_vec)          # Predict response
    return jsonify({'response': prediction[0]})    # Send the response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
