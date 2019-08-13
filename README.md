# Disaster Response Pipeline Project

It's a project in which we can classify several types of messages that appear during the time of a disaster, some of that types are: infrastructure, security, water, hospitals, first-aid, and many more.

Aditionally, we have the posibility to clean our messages (wrote by people during a disaster) and the categories (types of needs to cover), and later create a machine learning model to classify new emergency messages.

The application is developed using the Flask framework.

### Instructions:
1. Run the following command to install all the dependencies of the project.
    - `pip install -r requirements.txt`
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

### Application image:

![Application image](app.png?raw=true "Application image")