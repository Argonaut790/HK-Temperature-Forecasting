import mysql.connector
import matplotlib.pyplot as plot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

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
        "SELECT sun FROM temperature.join_result\
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
    #print(y)
    
    #split data into train set (70%) and test set (30%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    #--------------------------------------------------
    #building a polynomial regression model
    poly = PolynomialFeatures(degree=7)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)

    #fit data to the pr(polynomial regression) model
    pr_sun = LinearRegression()
    pr_sun.fit(x_train_poly, y_train)
    sun_train_ls = np.linspace(start=np.min(x_train), stop=np.max(x_train), num=100)
    sun_train_ls_poly = poly.fit_transform(sun_train_ls.reshape(-1,1))
    #--------------------------------------------------
    #linear regression model
    lr_sun = LinearRegression()
    lr_sun.fit(x_train, y_train)
    c_sun = lr_sun.intercept_
    m_sun = lr_sun.coef_
    #print(m_sun)
    #--------------------------------------------------
    #use the model to predict y value
    y_pred_train = lr_sun.predict(x_train) #lr model is done now reuse the x training data to predict it y value which is using 70% of the data
    y_pred_test = lr_sun.predict(x_test)
    y_pred_train_poly = pr_sun.predict(sun_train_ls_poly)
    #print(y_pred_train)

    # Sun-AvgTemp
    plot.subplot(1,3,1)
    plot.scatter(x_train, y_train, label="Training Data", color="red", s=8, alpha=0.7)
    plot.scatter(x_test, y_test, label="Testing Data", color="green", s=8, alpha=0.7)
    plot.plot(x_train, y_pred_train, label="Linear Regression", color='blue')
    plot.plot(sun_train_ls, y_pred_train_poly, label="Polynomial Regression", color='black')
    plot.legend()
    plot.title("Bright Sun Time - Average Temperature Relationship", y=1.04)
    plot.title(f"Linear: r^2={r2_score(y_train, y_pred_train):.2f}(Train), Test r^2={r2_score(y_test, y_pred_test):.2f}(Test)", loc='left', size=7)
    plot.title(f"Poly: r^2={r2_score(y_train, pr_sun.predict(x_train_poly)):.2f}(Train)", loc='right', size=7)
    plot.xlabel("Bright Sun Time")
    plot.ylabel("Average Temperature")
    
    #do the same for the rh(relative humidity)
    mycursor.execute(
        "SELECT rh FROM temperature.join_result\
        ORDER BY Year, Month, Day;"
    )

    x = mycursor.fetchall()
    #print(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    #polynomial regression
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)
    
    pr_rh = LinearRegression()
    pr_rh.fit(x_train_poly, y_train)
    rh_train_ls = np.linspace(20,100,100).reshape(-1,1)
    rh_train_ls_poly = poly.transform(rh_train_ls)

    lr_rh = LinearRegression()
    lr_rh.fit(x_train, y_train)
    c_rh = lr_rh.intercept_
    m_rh = lr_rh.coef_
    #print(m_rh)

    y_pred_train = lr_rh.predict(x_train) #lr model is done now reuse the x training data to predict it y value which is using 70% of the data
    y_pred_test = lr_rh.predict(x_test)
    y_pred_train_poly = pr_rh.predict(rh_train_ls_poly)
    #print(y_pred_train)

    # RH-AvgTemp
    plot.subplot(1,3,2)
    plot.scatter(x_train, y_train, label="Training Data", color="red", s=8, alpha=0.7)
    plot.scatter(x_test, y_test, label="Testing Data", color="green", s=8, alpha=0.7)
    plot.plot(x_train, y_pred_train, label="Linear Regression", color='blue')
    plot.plot(rh_train_ls, y_pred_train_poly, label="Polynomial Regression", color='black')
    plot.legend()
    plot.title("Relative Humidity - Average Temperature Relationship", y=1.04)
    plot.title(f"Linear: r^2={r2_score(y_train, y_pred_train):.2f}(Train), Test r^2={r2_score(y_test, y_pred_test):.2f}(Test)", loc='right', size=7)
    plot.xlabel("Relative Humidity")
    plot.ylabel("Average Temperature")
    
    #gsr part
    mycursor.execute(
        "SELECT gsr FROM temperature.join_result\
        ORDER BY Year, Month, Day;"
    )

    x = mycursor.fetchall()
    #print(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    #polynomial regression
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)
    
    pr_gsr = LinearRegression()
    pr_gsr.fit(x_train_poly, y_train)
    gsr_train_ls = np.linspace(0,30,100).reshape(-1,1)
    gsr_train_ls_poly = poly.transform(gsr_train_ls)

    lr_gsr = LinearRegression()
    lr_gsr.fit(x_train, y_train)
    c_gsr = lr_gsr.intercept_
    m_gsr = lr_gsr.coef_
    #print(m_gsr)

    y_pred_train = lr_gsr.predict(x_train) #lr model is done now reuse the x training data to predict it y value which is using 70% of the data
    y_pred_test = lr_gsr.predict(x_test)
    y_pred_train_poly = pr_gsr.predict(gsr_train_ls_poly)
    #print(y_pred_train)

    # GSR-AvgTemp
    plot.subplot(1,3,3)
    plot.scatter(x_train, y_train, label="Training Data", color="red", s=8, alpha=0.7)
    plot.scatter(x_test, y_test, label="Testing Data", color="green", s=8, alpha=0.7)
    plot.plot(x_train, y_pred_train, label="Linear Regression", color='blue')
    plot.plot(gsr_train_ls, y_pred_train_poly, label="Polynomial Regression", color='black')
    plot.legend()
    plot.title("Global Solar Radiation - Average Temperature Relationship", y=1.04)
    plot.title(f"Linear: r^2={r2_score(y_train, y_pred_train):.2f}(Train), Test r^2={r2_score(y_test, y_pred_test):.2f}(Test)", loc='right', size=7)
    plot.xlabel("Global Solar Radiation")
    plot.ylabel("Average Temperature")

    #show the plot
    plot.show()

if __name__ == "__main__":
    main()