# Disaster response pipeline

## Summary
This project is a first attemp to model the emergency messages in disaster situations and to classify it in some categories to prioritize the response of the emergency and rescue teams. The data was proportioned by Appen and consists in the messages registered in several emergency situations around the world, all of those translated into english. The model takes the text, lemmatize and vectorize it to do a TF-IDF transformer that join a Multi output classifier gives us the correspondent category of message.

## File descriptions
The data folder contains the data files containing the messages information: 
- disaster_categories.csv: the correspondent categories of messages in dataset.
- disaster_messages.csv: the emergency messages.
- process_data.py: the load, cleaning and saving data code.
  
The model folder contains the code to train and save the multi output classification model:
- train_classifier.py: the code to load, preprocess, train, evaluate and save the model.
- classifier.pkl: the model trined with the optimal parameters according to GridSearchCV.

The app folder contains the html and main python files to deploy the web application.

## Instructions:
1. Run the following commands in the project's root directory to set up the database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run the web app: `python run.py`
