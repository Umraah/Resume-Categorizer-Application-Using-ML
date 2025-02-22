import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample dataset (Replace this with actual data)
data = {
    "resume": [
        "Experienced Java Developer with Spring Boot",
        "Data Scientist proficient in Python and ML",
        "Manual and Automation Testing expert",
        "Frontend Web Developer skilled in HTML, CSS, JS",
        "HR professional with 5 years of experience",
    ],
    "category_id": [15, 6, 2, 24, 12]
}

df = pd.DataFrame(data)

# Function to clean resumes
def cleanResume(txt):
    txt = re.sub('http\S+\s', ' ', txt)  # Remove URLs
    txt = re.sub('RT|cc', ' ', txt)
    txt = re.sub('#\S+\s', ' ', txt)
    txt = re.sub('@\S+', '  ', txt)  
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)  # Remove non-ASCII
    txt = re.sub('\s+', ' ', txt)
    return txt.lower()

df['cleaned_resume'] = df['resume'].apply(cleanResume)

# Vectorization
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['cleaned_resume'])
y = df['category_id']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(tfidf_vectorizer, open("tfidf.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("Model and vectorizer saved successfully!")
