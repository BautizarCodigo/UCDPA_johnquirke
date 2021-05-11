import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf



from matplotlib import figure
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


class ProjectDetailAnalysis:

    def import_data(self):
        """Imports the data to be processed"""

        BASE_PATH = os.path.dirname(os.path.abspath(__file__))
        '''Import the data that will be analyse'''
        perth_prices = pd.read_csv(BASE_PATH + '/DATA/all_perth_310121.csv')

        return perth_prices

    def explore_the_data(self):
        '''Create reports that will give an understanding of the dataset
        run this function to check details'''

        df = self.import_data()

        print('*'*50)
        print('Housing Data Analysis Data Perth')
        print('*' * 50)
        print('Shape is ' + str(df.shape))
        print('*' * 50)
        print(df.info())
        print('*' * 50)
        print(df.describe().transpose())
        print('*' * 50)
        print('Check for missing values')
        print(df.isnull().sum())

    def explore_the_data(self):
        '''Explore the data using charts to help get a better
        understanding of the data'''
        df = self.import_data()
        sns.set_theme(style="whitegrid")

        '''Price'''
        #sns.displot(df, y='PRICE', height=5,aspect=2)

        #Analysis of Charts to find extra information and outliers
        ''' Boxplot '''
        #sns.boxplot(x=df["PRICE"])
        #sns.boxplot(x="BEDROOMS", y="PRICE", data=df).set_title('Number of BEDROOMS to PRICE')

        ''' Scatterplots'''
        #sns.scatterplot(x='PRICE', y='FLOOR_AREA', data=df)
        #sns.scatterplot(x='PRICE', y='LONGITUDE', data=df)
        #sns.scatterplot(x='PRICE', y='FLOOR_AREA', data=df)
        #sns.scatterplot(x='PRICE', y='LONGITUDE', data=df)
        #Shows the expensive areas
        #sns.scatterplot(x='LONGITUDE', y='LATITUDE', data=df, hue='PRICE')

        ''' Countplots'''
        #sns.countplot(df['BEDROOMS']).set_title('BEDROOMS in property')

        ''' Heatmap '''
        plt.figure(figsize=(12, 7))
        sns.heatmap(df.corr(), annot=True, cmap='viridis')
        plt.ylim(10, 0)
        plt.show()

    def data_leaning(self):
        """Clean the data, find missing data and replace or remove"""
        df = self.import_data()

        #Percentage of missing DATA in the DATA frame
        percentage_missing = 100 * df.isnull().sum() / len(df)

    def process_data(self):
        """Convert categorical variable into dummy/indicator variables."""

        df = self.import_data()

        #Use Dummies values for suburbs
        df2 = pd.get_dummies(df['SUBURB'])

        #Use the mean of Garages and Build to numbers for NULL values
        df.fillna(df.mean(), inplace=True)

        #Create a value if a property has a garage or not
        df['HAS_GARAGE'] = df['GARAGE'].apply(lambda x: 1 if x >= 11 or x <= 1 else 0)

        #New column for the month sold and cast to INT
        df['MONTH_SOLD'] = df['DATE_SOLD'].apply(lambda month: int(month[0:2]))

        # New column for the Year sold and cast to INT
        df['YEAR_SOLD'] = df['DATE_SOLD'].apply(lambda year: int(year[-5:]))


        #Drop Columns not needed anymore
        df.drop(labels=['ADDRESS', 'SUBURB', 'NEAREST_STN', 'NEAREST_SCH_RANK', 'DATE_SOLD', 'NEAREST_SCH'], axis=1, inplace=True)

        #Join up the new DATAFRAMES
        print(df.shape)
        dataframes = [df, df2]
        test_data = pd.concat(dataframes, axis=1)

        return test_data


    def supervised_testing_tensor(self):
        """Scaling and Train Test Split"""

        test_data = self.process_data()


        X = test_data.drop('PRICE', axis=1)
        y = test_data['PRICE']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        #Creating a Model

        model = Sequential()

        model.add(Dense(19, activation='relu'))
        model.add(Dense(19, activation='relu'))
        model.add(Dense(19, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse')

        model.fit(x=X_train, y=y_train.values,
                  validation_data=(X_test, y_test.values),
                  batch_size=128, epochs=100)

        #Convert losses to a Dataframe, loss and validation loss
        losses = pd.DataFrame(model.history.history)
        losses.plot()

        predictions = model.predict_classes(X_test)
        print(predictions.shape)
        print(y_test.shape)
        print()
        print(classification_report(y_test, predictions))
        print()
        print(confusion_matrix(y_test,predictions,  target_names='PRICE'))




        #Model Evaluation, Testing on a brand new house
        # single_house = test_data.drop('PRICE', axis=1).iloc[400]
        # single_house = scaler.transform(single_house.values.reshape(-1, 336))
        # print("*** Prediction " + str(model.predict(single_house)))
        # print("*** Actual Data of house " + str(test_data.iloc[400]['PRICE']))
        # print("*** Accuracy  " + str(test_data.iloc[400]['PRICE'] / model.predict(single_house)))
        #
        # #predictions = model.predict_classes(X_test)
        #
        # predictions = np.argmax(model.predict(X_test), axis=-1)
        # print(classification_report(y_test, predictions))

        #plt.show()







if __name__ == "__main__":
    hp = ProjectDetailAnalysis()
    hp.supervised_testing_tensor()


