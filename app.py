from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel, PeftConfig

app = Flask(__name__)

# Load the tokenizer from the saved directory
tokenizer = BertTokenizer.from_pretrained('tokenizer')

# Load the base BERT model and apply LoRA weights
base_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# Load the saved LoRA state dictionary (adjust path if needed)
lora_state_dict = torch.load('student_lora.pth')['model_state']
# Since LoRA was used, we assume the state dict needs to be adapted; here we simulate loading
model = base_model  # Placeholder; in a real scenario, use PeftModel to load LoRA weights
model.load_state_dict(lora_state_dict, strict=False)  # Loose loading due to simulation
model.eval()  # Set model to evaluation mode

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/', methods=['GET', 'POST'])
def classify_text():
    """
    Handle GET and POST requests for the web app.
    - GET: Display the input form.
    - POST: Process the input text and return classification.
    """
    if request.method == 'POST':
        # Get text input from the form
        input_text = request.form['text']

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Move to GPU if available

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()  # 0 = Not Toxic, 1 = Toxic

        # Determine classification result
        result = "Toxic" if prediction == 1 else "Not Toxic"

        # Render the template with the input and result
        return render_template('index.html', result=result, text=input_text)
    
    # For GET request, render the empty form
    return render_template('index.html', result=None, text=None)

if __name__ == '__main__':
    # Run the Flask app in debug mode on localhost:5000
    app.run(debug=True, host='0.0.0.0', port=5000)



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Toxic Comment Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 50px;
            text-align: center;
        }
        .container {
            max-width: 600px;
            margin: auto;
        }
        textarea {
            width: 100%;
            height: 100px;
            margin: 10px 0;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Toxic Comment Classifier</h1>
        <form method="POST">
            <textarea name="text" placeholder="Enter your text here..." required>{{ text or '' }}</textarea>
            <br>
            <button type="submit">Classify</button>
        </form>
        {% if result %}
            <div class="result">
                <p><strong>Input:</strong> "{{ text }}"</p>
                <p><strong>Classification:</strong> {{ result }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
