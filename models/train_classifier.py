import sys
import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from pandas import read_sql_table
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import create_engine

def load_data(database_filepath): 
    '''
    INPUT
    database_filepath: (str) the path where the database file is
    OUTPUT
    X: (pandas Series) the messages of the database
    Y: (pandas DataFrame) DataFrame which columns are the categories of the messages
    Y.columns: (list) the column names of Y
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, list(Y.columns)

def tokenize(text): 
    '''
    INPUT
    text: (str) the text to tokenize
    OUTPUT
    clean_tokens: (list) list of words tokenized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''    
    OUTPUT: the pipeline that process the text ready to use
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    pipeline.set_params(
        clf__estimator__max_depth=None, 
        clf__estimator__max_features='sqrt', 
        clf__estimator__min_samples_leaf=1
    )
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names): 
    '''
    INPUT
    model: (Scikit Pipeline) the model to evaluate
    X_test: (Pandas DataFrame) the features to evaluate prediction
    Y_test: (Pandas Series) the true values of the objective variable
    category_names: (list) the names of the categories of messages
    OUTPUT
    None (this function print the classification report of the prediction) 
    '''
    Y_predict = model.predict(X_test)
    for category in category_names: 
        i = category_names.index(category)
        print(classification_report(Y_test[category], Y_predict[: , i]))
    
def save_model(model, model_filepath): 
    '''
    INPUT
    model: (Scikit Pipeline) the model to save
    model_filepath: (str) the filepath where to save the model
    OUTPUT
    The pickle file with the model in the model_filepath given
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

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