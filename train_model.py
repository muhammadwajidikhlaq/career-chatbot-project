# Preprocess the Text
import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

#  Load dataset
df = pd.read_csv("career_guidance_dataset.csv")

#  Preprocessing
def preprocess(text):
    text = text.lower()     # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))    # Remove punctuation
    return text
# Clean Questions column 
df['Cleaned_Question'] = df['question'].apply(preprocess)

# Features and Labels
X_text = df['Cleaned_Question']
y = df['role']

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)      # Cleaned questions into numeric vector

# Check shape
print(" TF-IDF matrix shape:", X.shape)
print(" Labels shape:", y.shape)

# Save Vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")
print(" Vectorizer saved as vectorizer.pkl")

# Train Model
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load dataset
df = pd.read_csv("career_guidance_dataset.csv")

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "intent_model.pkl")
print(" Model as intent_model.pkl ")
