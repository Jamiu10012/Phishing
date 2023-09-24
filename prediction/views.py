# prediction/views.py


from django.shortcuts import render, redirect
import pickle
import urllib.parse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# Load the trained model
with open('model_pickle', 'rb') as model_file:
    model = pickle.load(model_file)


# Initialize the TF-IDF vectorizer without specifying max_features
vectorizer = TfidfVectorizer()


# Load your training data from a CSV file
training_data = pd.read_csv('C:/Users/AYEGORO/Documents/phishing_site_urls.csv')


# Extract the features from the training data
X_selected = training_data.drop(columns=['Label'], axis=1)
y = training_data['Label']


# Fit the vectorizer on the training data
vectorizer.fit(X_selected)


def index(request):
    return render(request, 'index.html')


def predict_phishing(url):
    # Encode the URL using UTF-8
    encoded_url = urllib.parse.quote(url, safe='')


    # Transform the URL input using the same vectorizer
    url_tfidf = vectorizer.transform([encoded_url])


    # Make predictions using the trained model
    prediction = model.predict(url_tfidf)


    return prediction


def result(request):
    if request.method == 'GET':
        url = request.GET.get('url', '')
       
        url = url

        count_i = url.count('/')
        count_c = url.count('.')


        # Ensure the URL is not empty

        if url:
            prediction = count_i >= 5
            predictionc = count_c >= 3
            
            if prediction:
                result_text = 'Phishing URL'
            elif predictionc:
                result_text = 'Phishing URL'
            else:
                result_text = 'Not a Phishing URL'
        
            # Make a prediction
        else:
            result_text = 'Please provide a URL for prediction.'

        # if url:
        #     prediction = predict_phishing(url)
       

        #     if prediction == "www.henkdeinumboomkwekerij.nl/language/pdf_fonts/smiles.php":
        #         result_text = 'Phishing URL'
        #     else:
        #         result_text = 'Not a Phishing URL'
        # else:
        #     result_text = 'Please provide a URL for prediction.'


        return render(request, 'result.html', {'result_text': result_text})
    else:
        return redirect('index')