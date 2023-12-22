import mysql.connector
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

def main() -> None:
    #retrieve data from mysql
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="#Tung266314",
        database="gsr"
    )
    mycursor = mydb.cursor()
    
    #run query, get the x xariable data
    mycursor.execute(
        "SELECT sun, rh, gsr FROM temperature.join_result\
        ORDER BY Year, Month, Day;"
    )

    #fetch data, get the dependent variable data
    x = mycursor.fetchall()
    #print(x)

    mycursor.execute(
        "SELECT avgTemp FROM temperature.join_result\
        ORDER BY Year, Month, Day;"
    )

    y = mycursor.fetchall()

    #split data into train set (70%) and test set (30%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    #<--------------------------------------------------------->

    # Select the best k features
    selector = SelectKBest(f_regression, k=3)
    x_new = selector.fit_transform(x_train, y_train)

    # Train a linear regression model on the selected features
    model = LinearRegression()
    model.fit(x_new, y_train)

    # Calculate the R-squared on the test set
    r_square = model.score(selector.transform(x_test), y_test)
    print(r_square)


if __name__ == "__main__":
    main()