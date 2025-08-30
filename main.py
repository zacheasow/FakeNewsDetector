import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report
import os
from tensorflow.keras.models import load_model

path = input("Enter the path of the WELFake_Dataset.csv")
df = pd.read_csv(path)


print(df.head())
print(df.info())
print(df['label'].value_counts())



df['combined_text'] = df['title'] + " " + df['text']
df['combined_text'] = df['combined_text'].fillna('')

# df = df.dropna(subset=['combined_text'])

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

df['cleaned_text'] = df['combined_text'].apply(clean_text)


X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')


X_train_tfidf = vectorizer.fit_transform(X_train)


X_test_tfidf = vectorizer.transform(X_test)


def create_model():

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    model.summary()


    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()


    history = model.fit(
        X_train_dense, y_train,
        epochs=6,
        batch_size=64,
        validation_split=0.2
    )

    loss, accuracy = model.evaluate(X_test_dense, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    model.save("fakenews.h5")
    return X_train_dense, X_test_dense

if not os.path.exists("fakenews.h5"):
    X_train_dense, X_test_dense = create_model()
else:
    model = load_model('fakenews.h5')

y_pred = model.predict(X_test_dense)
y_pred = (y_pred > 0.5).astype(int)



print(classification_report(y_test, y_pred))
def predict_fake_news(text):
    cleaned_text = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    text_dense = text_tfidf.toarray()
    prediction = model.predict(text_dense)
    confidence = prediction[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    return label, confidence

#TEST
# new_article = "This is a fake news article designed to mislead people."
# label, confidence = predict_fake_news(new_article)
# print(f"Prediction: {label} (Confidence: {confidence:.2f})")




import math
import requests
from bs4 import BeautifulSoup

def get_article_text(url):
    """Extracts and cleans main article text from a given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    article = soup.find("article")
    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = " ".join(p.get_text() for p in paragraphs if p.get_text().strip())

    return text if text else None 
while True:
    x = input("If you would like to check text from copy-paste, enter 1 \nIf you would like to check from a website link, enter 2.\nIf you are done, press 0 to exit. ")

    if x not in {"1", "2", "0"}:
        print("Invalid input.")
        continue

    if x == "1":
        new_article = input("Enter your article here: ")
        label, confidence = predict_fake_news(new_article)
        confidence = math.ceil(confidence * 10**5) / 10**5
        print(f"This is most likely {label.lower()} news, and I am {confidence * 100}% sure.")

    elif x == "2":
        url = input("Enter the website link here or press 0 to quit altogether: ")
        if url == "0":
            break

        article_text = get_article_text(url)
        if not article_text:
            print("Could not extract meaningful text from the URL. Try another website.")
            continue

        label, confidence = predict_fake_news(article_text)
        confidence = math.ceil(confidence * 10**5) / 10**5
        print(f"This is most likely {label.lower()} news")

    elif x == "0":
        break