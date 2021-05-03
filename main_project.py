import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from matplotlib import figure
from sklearn.model_selection import train_test_split



class ProjectDetailAnalysis:

    def import_data(self):

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
        '''Clean the data, find missing data and replace or remove'''
        df = self.import_data()

        #Percentage of missing DATA in the DATA frame
        percentage_missing = 100 * df.isnull().sum() / len(df)


    def process_data(self):
        '''Convert categorical variable into dummy/indicator variables.'''

        df = self.import_data()
        #Use Dummies values for suburbs
        df2 = pd.get_dummies(df['SUBURB'])


        #Use the mean of Garages and Build to numbers for NULL values
        df.fillna(df.mean(), inplace=True)

        #New column for the month sold and cast to INT
        df['MONTH_SOLD'] = df['DATE_SOLD'].apply(lambda month: int(month[0:2]))

        # New column for the Year sold and cast to INT
        df['YEAR_SOLD'] = df['DATE_SOLD'].apply(lambda year: int(year[-5:]))


        #Drop Columns not needed anymore
        df.drop(labels=['ADDRESS', 'SUBURB', 'NEAREST_STN', 'NEAREST_SCH_RANK', 'DATE_SOLD', 'NEAREST_SCH'], axis=1, inplace=True)

        #Join up the new DATAFRAMES
        dataframes = [df, df2]
        test_data = pd.concat(dataframes, axis=1)

        return test_data

    def train_testing(self):

        test_data = self.process_data()

        print(test_data.head())


if __name__ == "__main__":
    hp = ProjectDetailAnalysis()
    hp.train_testing()

