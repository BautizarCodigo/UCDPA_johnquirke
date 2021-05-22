import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf




from numpy import random
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier


from sklearn import metrics


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


class ProjectDetailAnalysis:

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

    def import_data(self):
        """Imports the data to be processed"""


        '''Import the data that will be analyse'''
        perth_prices = pd.read_csv(self.BASE_PATH + '/DATA/all_perth_310121.csv')


        return perth_prices

    def find_pattern_data(self):
        pass


    def explore_the_dataset(self):
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
        percentage_missing = 100 * df.isnull().sum() / len(df)
        print(percentage_missing)

    def graph_the_data(self):
        '''Explore the data using charts to help get a better
        understanding of the data'''
        df = self.import_data()
        sns.set_theme(style="whitegrid")


        '''Price'''
        #sns.displot(df, y='PRICE', height=5,aspect=2)

        #Analysis of Charts to find extra information and outliers
        ''' Boxplot '''
        # sns.boxplot(x=df["PRICE"])
        # sns.boxplot(x="BEDROOMS", y="PRICE", data=df).set_title('Number of BEDROOMS to PRICE')

        ''' Scatterplots'''
        # sns.scatterplot(x='PRICE', y='FLOOR_AREA', data=df)
        # sns.scatterplot(x='PRICE', y='LONGITUDE', data=df)
        # sns.scatterplot(x='PRICE', y='LONGITUDE', data=df)

        #Shows the expensive areas
        #sns.set(rc={'figure.figsize': (5, 7)})
        #sns.scatterplot(data=df, x='LONGITUDE', y='LATITUDE', hue='PRICE', style="PRICE")
        #sns.scatterplot(data=df, x='LONGITUDE', y='LATITUDE', hue='PRICE',s=10,marker="+")


        # ''' Countplots'''
        #sns.countplot(df['BEDROOMS']).set_title('BEDROOMS in property')

        ''' Heatmap '''
        # plt.figure(figsize=(12, 7))
        # sns.heatmap(df.corr(), annot=True, cmap='viridis')
        #plt.ylim(10, 0)

        plt.show()

    def process_data(self):
        """Convert categorical variable into dummy/indicator variables."""

        # Map the weekly income to the df['SUBURBS']
        df = self.import_data()
        income = pd.read_csv('DATA/suburb_Weekly_income.csv')
        income_index = income.set_index('Suburb')
        income_dict = income_index['Weekly Income'].to_dict()
        df['INCOME_SUBURB'] = df['SUBURB'].map(income_dict)




        # Use Dummies values for suburbs
        df2 = pd.get_dummies(df['SUBURB'])

        # Fill missing values
        df['BUILD_YEAR'] = df['BUILD_YEAR'].fillna(df['BUILD_YEAR'].median())

        # Create a value if a property has a garage or not
        #df['HAS_GARAGE'] = df['GARAGE'].apply(lambda x: 1 if x >= 11 or x <= 1 else 0)
        df['GARAGE'] = df['GARAGE'].fillna(0)  # fill missing data with 0

        # New column for the month sold and cast to INT
        df['MONTH_SOLD'] = df['DATE_SOLD'].apply(lambda month: int(month[0:2]))

        # New column for the Year sold and cast to INT
        df['YEAR_SOLD'] = df['DATE_SOLD'].apply(lambda year: int(year[-5:]))

        # Use the mean of Build to numbers for NULL values
        df.fillna(df.mean(), inplace=True)

        # Drop Columns not needed anymore
        df.drop(labels=['ADDRESS', 'SUBURB', 'NEAREST_STN', 'NEAREST_SCH_RANK', 'DATE_SOLD', 'NEAREST_SCH'], axis=1,
                inplace=True)

        #Remove outliers
        # z_scores = stats.zscore(df)
        # df = df[(z_scores < 3).all(axis=1)]

        # Join up the new DATAFRAMES
        dataframes = [df, df2]
        test_data = pd.concat(dataframes, axis=1)

        return test_data


    def  variance_elements(self):
        '''Find the elements that effect the price the most'''
        test_data = self.process_data()

        X = test_data.drop('PRICE', axis=1).values
        y = test_data['PRICE'].values

        elements = test_data.drop('PRICE', axis=1).columns
        lasso = Lasso(alpha=1)
        lasso_coef = lasso.fit(X, y).coef_
        _ = plt.plot(range(len(elements)), lasso_coef)
        _ = plt.xticks(range(len(elements)), elements, rotation=60)

        plt.show()


    def linear_regression_testing(self):

        test_data = self.process_data()

        X = test_data.drop('PRICE', axis=1).values
        y = test_data['PRICE'].values

        # splitting Train and Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

        # standardization scaler - fit&transform on train, fit only on test
        s_scaler = MinMaxScaler()
        X_train = s_scaler.fit_transform(X_train.astype(np.float))
        X_test = s_scaler.transform(X_test.astype(np.float))

        # Multiple Liner Regression
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # evaluate the model (intercept and slope)
        print(regressor.intercept_)
        print(regressor.coef_)

        # predicting the test set result
        y_pred = regressor.predict(X_test)

        # put results as a DataFrame
        coeff_df = pd.DataFrame(regressor.coef_, test_data.drop('PRICE', axis=1).columns, columns=['Coefficient'])
        print(coeff_df)

        # visualizing residuals
        fig = plt.figure(figsize=(10, 5))
        residuals = (y_test - y_pred)
        sns.distplot(residuals)
        plt.show()

        # compare actual output values with predicted values
        y_pred = regressor.predict(X_test)
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        df1 = df.head(10)
        print(df1)
        # evaluate the performance of the algorithm (MAE - MSE - RMSE)
        from sklearn import metrics
        print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
        print('MSE:', metrics.mean_squared_error(y_test, y_pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('VarScore:', metrics.explained_variance_score(y_test, y_pred))

    def keras_regression(self):
        """Scaling and Train Test Split"""

        test_data = self.process_data()
        X = test_data.drop('PRICE', axis=1)
        y = test_data['PRICE']

        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imp.fit(X)
        # X = imp.transform(X)

        #Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        #Scale the set
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        #Define Sequential model with 5 layers

        model = keras.Sequential(
            [
            layers.Dense(336, activation='relu', name="layer1"),
            layers.Dense(168, activation='relu', name="layer2"),
            layers.Dense(84, activation='relu', name="layer3"),
            layers.Dense(42, activation='relu', name="layer4"),
            layers.Dense(21, activation='relu', name="layer5"),
            ]
        )

           # output layer
        model.add(Dense(1))

        # Compile model
        model.compile(optimizer='adam', loss='mse')

        #Stop so not to overfit
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

        model.fit(x=X_train, y=y_train,
                  validation_data=(X_test, y_test),
                  batch_size=128, epochs=600,
                  verbose=1,
                  callbacks=[early_stop])

        #Convert losses to a Dataframe, loss and validation loss
        losses = pd.DataFrame(model.history.history)
        #print(losses)
        losses.plot()
        #plt.show()

        y_pred = model.predict(X_test)
        model.pop()
        print(len(model.layers))  # 2
        model.summary()

        #Get a random number to test
        random_test = x = random.randint(30000)
        print(random_test)

        #Model Evaluation, Testing on a brand new house
        single_house = test_data.drop('PRICE', axis=1).iloc[random_test]
        single_house = scaler.transform(single_house.values.reshape(-1, 336))
        prediction = model.predict(single_house)[0][0]
        actual_price = test_data.iloc[random_test]['PRICE']
        percent_of_actual = test_data.iloc[random_test]['PRICE'] / model.predict(single_house)[0][0]
        percent_from_actual = (actual_price - prediction) / actual_price


        print("Prediction           : " + str(prediction))
        print("Actual Price of house: " + str(actual_price))
        print("Accuracy             :  " + str(percent_of_actual))
        print("% Difference from Actual  :  " + str(percent_from_actual))
        print()

        print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
        print('MSE:', metrics.mean_squared_error(y_test, y_pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('VarScore:', metrics.explained_variance_score(y_test, y_pred))




        return random_test, prediction, actual_price, percent_of_actual, percent_from_actual

    def graph_results(self):

        #Run a Series of tests
        num_of_tests = int(input('How many tests do you want to run? '))

        while num_of_tests > 1:
            #unpack the tuple
            house_row, prediction, actual_price, percent_of_actual, percent_from_actual = self.keras_regression()


            with open(self.BASE_PATH + '/DATA/model_results.csv', 'a', newline='') as results_file:
                results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                results_writer.writerow([house_row, prediction, actual_price, percent_of_actual, percent_from_actual])

            num_of_tests = num_of_tests -1




if __name__ == "__main__":
    hp = ProjectDetailAnalysis()
    #hp.process_data()
    #hp.graph_the_data()
    #hp.supervised_testing_tensor()
    #hp.linear_regression_testing()
    #hp.keras_regression()
    #hp.regression_small()
    #hp.variance_elements()
    #hp.k_nearest_neighbours()
    hp.graph_results()

