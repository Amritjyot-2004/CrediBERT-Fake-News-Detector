from flask import Flask, request, render_template, jsonify, url_for
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

model_path = "D:/Projects/FND/Detector"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

model.eval()

bot = []
user = []

news_map = {0:"Real", 1:"Fake"}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ['POST'])
def predict():
    headline = request.form['headline']
    user.append("User: "+headline+"\n")
    
    inputs = tokenizer(headline, return_tensors = 'pt', padding = True, truncation = True, max_length = 32)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    bot.append(f"CrediBERT: This news is {news_map[predicted_class]}\n\n")
    return render_template("index.html",pred_text = f"This news is {news_map[predicted_class]}")

@app.route("/history", methods = ['POST'])
def history():
    chat_history = ""
    chat_history_html = ""
    for i in range(len(bot)):
        chat_history+=user[i]+"\n"+bot[i]+"\n\n"
        chat_history_html = chat_history.replace("\n", "<br>")  # Convert newlines to <br>
    return render_template("index.html", history=chat_history_html)

#for direct api calls
@app.route("/predict_api")
def predict_api():
    data = request.get_json(force = True)

    inputs = tokenizer(data, return_tensors = 'pt', padding = True, truncation = True, max_length = 32)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    output = f"This news is {news_map[predicted_class]}"
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8080)