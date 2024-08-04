# import libraries
import sys
import numpy as np
import pandas as pd
from pandas import read_csv
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath): 
    '''
    INPUT
    messages_filepath: (str) the filepath of the .csv file with messages data
    categories_filepath: (str) the filepath of the .csv file with the categories
    OUTPUT
    df: (Pandas DataFrame) the DataFrame with the messages and the categories merged
    '''
    messages = read_csv(messages_filepath)
    categories = read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on='id')
    return df

def clean_data(df): 
    '''
    INPUT
    df: (Pandas DataFrame) the DataFrame which contains messages and categories uncleaned
    OUTPUT
    df: (Pandas DataFrame) the DataFrame cleaned, the categories are now integer columns
    '''
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories: 
        categories[column] = categories[column].apply(lambda x: x[-1]).astype(int)
        
    categories = categories[categories['related'] != 2]
        
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename): 
    '''
    INPUT
    df: (Pandas DataFrame) the cleaned DataFrame to save
    database_filename: (str) the name to give to df as database
    OUPUT
    This function saves as database the cleaned DataFrame in the given filepath
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')

def main(): 
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
        
if __name__ == '__main__':
    main()