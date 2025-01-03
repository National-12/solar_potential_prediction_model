from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained ML model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/")
def home():
    return "ML Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input JSON data
        data = request.json
        # Example: Convert data into a numpy array (adjust based on your model's input format)
        input_features = np.array(data["features"]).reshape(1, -1)
        
        # Get predictions
        prediction = model.predict(input_features)
        
        # Return the prediction as a response
        result = {"prediction": prediction.tolist()}
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()
