from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Define a set of human-written and AI-generated documents
human_documents,ai_documents = [],[]

 
try:
    i=1
    while(True):
        ai_documents.append(open(f'AI_text/{i}.txt').read().lower())
        human_documents.append(open(f'Human_text/{i}.txt').read().lower())
        i+=1
except Exception as e:
    pass



# labels = ['human'] * len(human_documents) + ['AI'] * len(ai_documents)


# vectorizer = TfidfVectorizer()

# # Fit the vectorizer to the documents
# vectorizer.fit(documents)

# # Extract features from the documents
# features = vectorizer.transform(documents)

# # Split the feature matrix and labels into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features.toarray(), labels, test_size=0.2)

# # Train a Random Forest classifier on the training data
# rf_classifier = RandomForestClassifier()
# rf_classifier.fit(X_train, y_train)

# # Test the classifier on the testing data
# predictions = rf_classifier.predict(X_test)

# # Print the accuracy of the classifier
# accuracy = sum(predictions == y_test) / len(y_test)
# print("Accuracy:", accuracy)