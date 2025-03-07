from flask import Flask, render_template, request, jsonify
import pickle
import re
import pytesseract
from PIL import Image
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import logging
import csv

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Loading the saved model and vectorizer
with open('fake_news_detector.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing the text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Remove all extra whitespaces
    text = text.lower()  # Convert text to lowercase
    text = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    return ' '.join(text)

# Function to predict if the news is real or fake
def predict_news(news_text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(news_text)

    # Transform the preprocessed text using the TF-IDF vectorizer
    transformed_text = vectorizer.transform([preprocessed_text])

    # Make a prediction
    prediction = model.predict(transformed_text)

    # Return the prediction
    return 'Real' if prediction[0] == 1 else 'Fake'

# Route to render the home page
@app.route("/")
def my_home():
    return render_template('index.html')

# Route to render other HTML pages
@app.route("/<string:page_name>")
def html_page(page_name):
    return render_template(page_name)

# Route to handle text input prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'news_text' in request.form:
            news_text = request.form['news_text']
            app.logger.debug(f"Received text input: {news_text}")
            prediction = predict_news(news_text)
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'No text input provided.'})
    except Exception as e:
        app.logger.error(f"Error processing text input: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'})

# Route to handle image input prediction
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            app.logger.debug(f"Received image file: {file.filename}")

            # Ensure the uploads directory exists
            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            # Save the file to a temporary location
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Open and process the image
            try:
                image = Image.open(file_path)
                news_text = pytesseract.image_to_string(image)
                app.logger.debug(f"Extracted text from image: {news_text}")
            except Exception as e:
                app.logger.error(f"Error opening image: {e}")
                return jsonify({'error': 'An error occurred while processing the image.'})

            # Check if text is extracted before prediction
            if news_text:
                prediction = predict_news(news_text)
                return jsonify({'prediction': prediction, 'extracted_text': news_text})
            else:
                return jsonify({'error': 'No text found in the image'})

            # Clean up the uploaded file
            os.remove(file_path)
        else:
            return jsonify({'error': 'File upload failed'})

    except Exception as e:
        app.logger.error(f"Error processing image input: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'})

# Route to handle contact form submission
@app.route('/submit_form', methods=['POST'])
def submit_form():
    try:
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']
        app.logger.debug(f"Received form data: email={email}, subject={subject}, message={message}")

        # Write form data to CSV file
        with open('database.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([email, subject, message])

        return render_template('contact.html', success=True)
    except Exception as e:
        app.logger.error(f"Error saving form data: {e}")
        return render_template('contact.html', success=False)