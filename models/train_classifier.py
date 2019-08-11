import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

import nltk
import pandas as pd
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    engine = create_engine('sqlite:///%s'%(database_filepath))
    df = pd.read_sql_table("Messages", con=engine)
    X = df.message.values
    Y_raw = df[list(set(df.columns) - set(["id", "message", "original", "genre"]))]
    Y = Y_raw.copy()
    Y[Y>1] = 1
    return X, Y, Y.columns

def tokenize(text):
    stop_words = nltk.corpus.stopwords.words('english')
    
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)

    tokens = [word for word in tokens if not(word in stop_words)]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1)))
    ])

    parameters = {
        'vect__min_df': [5],
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 50],
        'clf__estimator__min_samples_split':[10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))

def save_model(model, model_filepath):
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()