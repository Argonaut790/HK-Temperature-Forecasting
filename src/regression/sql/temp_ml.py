import mysql.connector
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def main() -> None:
    #retrieve data from mysql
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="#Tung266314",
        database="gsr"
    )
    mycursor = mydb.cursor()
    
    mycursor.execute(
        "SELECT sun, rh, gsr FROM temperature.join_result\
        ORDER BY Year, Month, Day;"
    )

    x = mycursor.fetchall()
    #print(x)

    mycursor.execute(
        "SELECT avgTemp FROM temperature.join_result\
        ORDER BY Year, Month, Day;"
    )

    y = mycursor.fetchall()
    #print(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    #scaler the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    #print(x_train_scaled)

    lr = LinearRegression()
    lr.fit(x_train_scaled, y_train)
    c = lr.intercept_
    m = lr.coef_
    #print(m)

    y_pred_train = lr.predict(x_train_scaled) #lr model is done now reuse the x training data to predict it y value which is using 70% of the data
    #print(y_pred_train)

    plot.subplot(1,2,1)
    plot.scatter(y_train, y_pred_train, s=1)
    plot.title("Average Temperature - King's Park (Train Data)", y=1.04)
    plot.title("*Prediction based on sun time, relative humidity and global solar radiation", loc='left', size=7)
    plot.title(f"r^2 = {r2_score(y_train, y_pred_train):.2f}", loc='right', size=7)
    plot.xlabel("Actualy Average Temperature")
    plot.ylabel("Predicted Average Temperature")

    #using test data
    y_pred_test = lr.predict(x_test_scaled)
    plot.subplot(1,2,2)
    plot.scatter(y_test, y_pred_test, s=1)
    plot.title("Average Temperature - King's Park (Test Data)", y=1.04)
    plot.title("*Prediction based on sun time, relative humidity and global solar radiation", loc='left', size=7)
    plot.title(f"r^2 = {r2_score(y_test, y_pred_test):.2f}", loc='right', size=7)
    plot.xlabel("Actualy Average Temperature")
    plot.ylabel("Predicted Average Temperature")
    plot.show()

if __name__ == "__main__":
    main()