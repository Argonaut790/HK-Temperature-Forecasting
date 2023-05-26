import pandas as pd
import matplotlib.pyplot as plt
import os
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


AVG_TEMP_DATA = os.getcwd() + "\data\CLMTEMP_KP_.csv"
GSR_DATA = os.getcwd() + "\data\daily_KP_GSR_ALL.csv"
RH_DATA = os.getcwd() + "\data\daily_KP_RH_ALL.csv"
SUN_DATA = os.getcwd() + "\data\daily_KP_SUN_ALL.csv"
RF_DATA = os.getcwd() + "\data\daily_KP_RF_ALL.csv"
UV_DATA = os.getcwd() + "\data\daily_KP_UV_ALL.csv"
WSPD_DATA = os.getcwd() + "\data\daily_KP_WSPD_ALL.csv"

DEGREE = 6

def DataJoinning(AVG_TEMP_df: pd.DataFrame, GSR_df: pd.DataFrame, RH_df: pd.DataFrame, SUN_df: pd.DataFrame, RF_df: pd.DataFrame, UV_df: pd.DataFrame, WSPD_df: pd.DataFrame):
    AVG_TEMP_df.rename(columns={'date':'Date' ,'Value':'AVG_TEMP', 'data Completeness':'AVG_TEMP_Completeness'}, inplace=True)
    GSR_df.rename(columns={'date':'GSR_date' ,'Value':'GSR', 'data Completeness':'GSR_Completeness'}, inplace=True)
    RH_df.rename(columns={'date':'RH_date' ,'Value':'RH', 'data Completeness':'RH_Completeness'}, inplace=True)
    SUN_df.rename(columns={'date':'SUN_date' ,'Value':'SUN', 'data Completeness':'SUN_Completeness'}, inplace=True)
    RF_df.rename(columns={'date':'RF_date' ,'Value':'RF', 'data Completeness':'RF_Completeness'}, inplace=True)
    UV_df.rename(columns={'date':'UV_date' ,'Value':'UV', 'data Completeness':'UV_Completeness'}, inplace=True)
    WSPD_df.rename(columns={'date':'WSPD_date' ,'Value':'WSPD', 'data Completeness':'WSPD_Completeness'}, inplace=True)

    # Drop rows with missing data
    AVG_TEMP_df = AVG_TEMP_df[AVG_TEMP_df['AVG_TEMP_Completeness'] == 'C']
    GSR_df = GSR_df[GSR_df['GSR_Completeness'] == 'C']
    RH_df = RH_df[RH_df['RH_Completeness'] == 'C']
    SUN_df = SUN_df[SUN_df['SUN_Completeness'] == 'C']
    RF_df = RF_df[RF_df['RF_Completeness'] == 'C']
    UV_df = UV_df[UV_df['UV_Completeness'] == 'C']
    WSPD_df = WSPD_df[WSPD_df['WSPD_Completeness'] == 'C']

    dfs = [AVG_TEMP_df, GSR_df, RH_df, SUN_df, RF_df, UV_df, WSPD_df]
    # dfs = [df.set_index('date') for df in dfs]
    # join_df = dfs[0].join(dfs[1:])
    join_df = pd.concat(dfs, axis=1, join='inner')
    join_df = join_df.drop(['GSR_date', 'RH_date', 'SUN_date', 'RF_date', 'UV_date', 'WSPD_date', 
                            'AVG_TEMP_Completeness', 'GSR_Completeness', 'RH_Completeness', 'SUN_Completeness', 'RF_Completeness', 'UV_Completeness', 'WSPD_Completeness'], axis=1)
    return join_df

def DataPreprocessing(df: pd.DataFrame):
    
    df = df.drop('Date', axis=1)
    df = df.astype(float)
    """
    # Plot the relationship graph
    for column in df.columns:
        if column == 'GSR':
            continue
        plt.scatter(df[column], df['GSR'], s=1)
        plt.xlabel(column)
        plt.ylabel('GSR')
        plt.show()
    """

    # apply normalization techniques
    for column in df.columns:
        try:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
        except:
            logger.error(column + " something wrong here")
            continue

    # Get the correlation matrix
    corr = df.corr()
    logger.info(corr)

    # Seperate x and y df
    x_df = df.drop(['GSR', 'RH', 'RF', 'WSPD'], axis=1)
    y_df = df['GSR']
    # logger.info(x_df)
    # logger.info(y_df)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

def FindingOptimalDegree(x_train, x_test, y_train, y_test):
    # check our accuracy for each degree, the lower the error the better!
    number_degrees = range(1, 9)
    plt_r2_train = []
    plt_r2_test = []
    for degree in number_degrees:

        poly_model = PolynomialFeatures(degree=degree)
        
        x_train_poly = poly_model.fit_transform(x_train)
        x_test_poly = poly_model.fit_transform(x_test)
        poly_model.fit(x_train_poly, y_train)
        
        regression_model = LinearRegression()
        regression_model.fit(x_train_poly, y_train)
        y_train_predicted = regression_model.predict(x_train_poly)
        y_test_predict = regression_model.predict(x_test_poly)

        plt_r2_train.append(r2_score(y_train, y_train_predicted))
        plt_r2_test.append(r2_score(y_test, y_test_predict))
    
    plt.scatter(number_degrees,plt_r2_train, color="green")
    plt.plot(number_degrees,plt_r2_train, color="blue") 
    plt.scatter(number_degrees,plt_r2_test, color="gray")
    plt.plot(number_degrees,plt_r2_test, color="red")
    plt.xlabel("Degree")
    plt.ylabel("r2 score")
    plt.title("r2 score for each degree in GSR vs AVG_TEMP, SUN, UV")
    plt.legend(["Train", "Train", "Test", "Test"])
    plt.show()

def PolynomialRegressionModel(x_train, x_test, y_train, y_test):
    poly_model = PolynomialFeatures(degree=DEGREE)
    x_train_poly = poly_model.fit_transform(x_train)
    x_test_poly = poly_model.fit_transform(x_test)
    poly_model.fit(x_train_poly, y_train)
    lr = LinearRegression()
    lr.fit(x_train_poly, y_train)
    slope = lr.coef_
    intercept = lr.intercept_
    # logger.info("Slope: {}".format(slope))
    # logger.info("Intercept: {}".format(intercept))
    
    # Train Model
    poly_model.fit(x_train_poly, y_train)
    # Use Linear Regression as the base
    regression_model = LinearRegression()
    regression_model.fit(x_train_poly, y_train)
    # Predict
    y_train_predicted = regression_model.predict(x_train_poly)
    y_test_predict = regression_model.predict(x_test_poly)

    # Plot the results -Train
    plt.subplot(1,2,1)
    r2_train = r2_score(y_train, y_train_predicted)
    plt.scatter(y_train, y_train_predicted, s=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("GSR Actual vs Predicted (Training) AVG_TEMP, SUN, UV", y=1.04)
    plt.title(f"R2: {r2_train:.4f}", loc='right', size=7)
    plt.title(f"Degree: {DEGREE}", loc='left', size=7)
    
    # Plot the results -Test
    plt.subplot(1,2,2)
    r2_test = r2_score(y_test, y_test_predict)
    plt.scatter(y_test, y_test_predict, s=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("GSR Actual vs Predicted (Testing) AVG_TEMP, SUN, UV", y=1.04)
    plt.title(f"R2: {r2_test:.4f}", loc='right', size=7)
    plt.title(f"Degree: {DEGREE}", loc='left', size=7)
    plt.show()

def LinearRegressionModel(x_train, x_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    slope = lr.coef_
    intercept = lr.intercept_
    logger.info("Slope: {}".format(slope))
    logger.info("Intercept: {}".format(intercept))
    
    # Plot the results -Train
    plt.subplot(1,2,1)
    y_pred_train = lr.predict(x_train)
    r2_train = r2_score(y_train, y_pred_train)
    plt.scatter(y_train, y_pred_train, s=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (Training)", y=1.04)
    plt.title(f"R2: {r2_train:.4f}", loc='right', size=7)
    
    # Plot the results -Test
    plt.subplot(1,2,2)
    y_pred_test = lr.predict(x_test)
    r2_test = r2_score(y_test, y_pred_test)
    plt.scatter(y_test, y_pred_test, s=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (Testing)", y=1.04)
    plt.title(f"R2: {r2_test:.4f}", loc='right', size=7)
    plt.show()


def main():
    try:
        AVG_TEMP_df = pd.read_csv(AVG_TEMP_DATA, parse_dates={'date':['Year', 'Month', 'Day']}, skiprows=[0,1])
        GSR_df = pd.read_csv(GSR_DATA, parse_dates={'date':['Year', 'Month', 'Day']}, skiprows=[0,1])
        RH_df = pd.read_csv(RH_DATA, parse_dates={'date':['Year', 'Month', 'Day']}, skiprows=[0,1])
        SUN_df = pd.read_csv(SUN_DATA, parse_dates={'date':['Year', 'Month', 'Day']}, skiprows=[0,1])
        RF_df = pd.read_csv(RF_DATA, parse_dates={'date':['Year', 'Month', 'Day']}, skiprows=[0,1])
        UV_df = pd.read_csv(UV_DATA, parse_dates={'date':['Year', 'Month', 'Day']}, skiprows=[0,1])
        WSPD_df = pd.read_csv(WSPD_DATA, parse_dates={'date':['Year', 'Month', 'Day']}, skiprows=[0,1])
    except:
        logger.error("Error reading data files")
        return

    join_df = DataJoinning(AVG_TEMP_df, GSR_df, RH_df, SUN_df, RF_df, UV_df, WSPD_df)
    logger.info(join_df.info())
    logger.info(join_df.isnull().sum())
    x_train, x_test, y_train, y_test = DataPreprocessing(join_df)
    # FindingOptimalDegree(x_train, x_test, y_train, y_test)
    PolynomialRegressionModel(x_train, x_test, y_train, y_test)
    # LinearRegressionModel(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()