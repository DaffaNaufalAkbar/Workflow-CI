import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

mlflow.autolog()

# Load Data
df = pd.read_csv('tokopedia_preprocessing.csv')

X = df['review_text']
y = df['label_sentimen']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlflow.set_experiment("Sentimen_Tokopedia_Autolog")

with mlflow.start_run(run_name="Pipeline_NB_Tfidf"):
    
    # Pipeline
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('nb', MultinomialNB())
    ])

    # Training Model
    model_pipeline.fit(X_train, y_train)

    # Prediksi dan Evaluasi
    y_pred = model_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f'Accuracy Score: {acc:.2%}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    mlflow.log_metric("test_accuracy", acc)