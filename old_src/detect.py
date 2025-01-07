import json
import math
import os

from vectorizer import CharFrequencyVectorizer
from src.naive_bayes import NaiveBayesBinaryClassifier

class nattydetect:
    def __init__(self, vectorizer_path="models/vectorizer.json", model_path="models/naive_bayes.json"):
        # Load the vectorizer
        self.vectorizer = CharFrequencyVectorizer()
        self.vectorizer.load(vectorizer_path)

        # Load the model
        self.model = NaiveBayesBinaryClassifier()
        self.model.load(model_path)

    def detect(self, text):
        """
        Vectorize the text and classify it.
        Return True if natural language (class 0), False if code (class 1).
        """
        freq_vector = self.vectorizer.transform(text)
        pred_class = self.model.predict(freq_vector)
        # pred_class will be 0 for natural language, 1 for code
        return (pred_class == 0)
    
# Example usage:
detector = nattydetect()
is_natural = detector.detect('''

The application deadline is Sunday, December 15 for the workshop on January 17-20, 2025 in Boston, MA. Since admissions are decided on a rolling basis, we recommend you apply as soon as possible.

We are only able to accept those who will be over 18 at the time of the workshop. By attending the workshop, you agree to complete a 30-minute post-workshop feedback survey a few months after the event.

If you’re not sure whether you are a good fit for the workshops, we encourage you to apply anyway. Some students who found our events very useful have told us they wouldn’t have applied unless we had encouraged them to do so.

                             ''')
print(is_natural)  # True if classified as natural language, False otherwise