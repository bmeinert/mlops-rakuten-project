import re
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

STOP_WORDS = set(stopwords.words("english")) | \
             set(stopwords.words("french")) | \
             set([
                 "chez", "der", "plu", "haut", "peut", "non", "100", "produit",
                 "lot", "tout", "cet", "cest", "sou", "san"
             ])

lemmatizer = WordNetLemmatizer()

class ProductTypePredictorMLflow:
    def __init__(self, model, vectorizer_path, product_dictionary_path):
        self.model = model
        self.stop_words = STOP_WORDS
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = joblib.load(f)
        with open(product_dictionary_path, "rb") as f:
            self.product_dictionary = joblib.load(f)

    def preprocess(self, designation, description=""):
        cleaned = self.clean_text_static(designation, description)
        return self.vectorizer_transform(pd.Series([cleaned]), self.vectorizer)

    def predict(self, designation, description=""):
        if not isinstance(designation, str):
            raise ValueError("designation has to be a string.")
        if not isinstance(description, str):
            raise ValueError("description has to be a string.")
        
        vectorized = self.preprocess(designation, description)
        prediction = self.model.predict(vectorized)[0]
        return self.product_dictionary[int(prediction)]

    @staticmethod
    def clean_text_static(designation, description):
        """Only text cleaning, no vectorization"""
        if pd.isna(description):
            description = ""
        text = f"{designation} {description}"
        if not isinstance(text, str):
            raise ValueError("text has to be a string.")
        # Remove special characters and lowercase the text
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        # Tokenize the text
        tokens = word_tokenize(text)
        # Lemmatize tokens
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        #Remove stopwords
        filtered_tokens = [token for token in lemmas if token not in STOP_WORDS]
        # Returns the cleaned text as a string
        return " ".join(filtered_tokens)
    
    @staticmethod
    def vectorizer_transform(series, vectorizer):
        """Only vectorization, no text cleaning"""
        if not isinstance(series, pd.Series):
            raise ValueError("series has to be a pandas Series.")
        return vectorizer.transform(series)
    