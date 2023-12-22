from turtle import color
import mysql.connector
import matplotlib.pyplot as plot
import numpy as np
from sklearn.metrics import r2_score

def main()->None:
    #retrieve data from mysql
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="#Tung266314",
        database="gsr"
    )
    mycursor = mydb.cursor()
    mycursor.execute(
        "select year, avg(value) as avg, max(value) as max, min(value) as min\
        from gsr.daily_kp_gsr_all \
        group by year \
        order by year"
    )
    result_2020 = mycursor.fetchall()
    #turn data to x,y
    year = []
    avg = []
    max = []
    min = []

    for data in result_2020:
        year.append(data[0])
        avg.append(data[1])
        max.append(data[2])
        min.append(data[3])

    avg_model = np.poly1d(np.polyfit(year, avg, 3))
    max_model = np.poly1d(np.polyfit(year, max, 3))
    '''
    #polyfit(x, y, deg) -> parameter[]
    polyfit turn x,y point into a degree of 3 equation, 
    -> parameter[0] * x^3 + parameter[1] * x^2 + ...
    it will return 4 constant which are parameter[0:4]
    #poly1d(list) -> porameter to equation
    poly1d turn those parameter into a equation, which is
    parameter[0] * x^3 + parameter[1] * x^2 + ...
    '''
    line_space_100 = np.linspace(1992, 2021, 100)
    '''
    #linspace(str_pt, end_pt, spaces)
    linespace separate start point to end point into 100 spaces
    [1992, 1992.29292929, 1992.58585859 ...]
    '''
    year = np.array(year)
    avg = np.array(avg)
    max = np.array(max)
    min = np.array(min)

    #plot 1
    plot.subplot(1,2,1)
    plot.scatter(year, avg)
    plot.plot(line_space_100, avg_model(line_space_100), color="green")
    plot.title("Global Solar Radiation Trend - King's Park")
    plot.title(f"r = {r2_score(avg, avg_model(year))*100:.2f}%", loc = "right", size = 7)
    plot.xlabel("Year (1992-2021)")
    plot.ylabel("Average Solar Radiation (MJ/m^2)")
    #plot 2
    plot.subplot(1,2,2)
    plot.scatter(year, max)
    plot.plot(line_space_100, max_model(line_space_100), color="green")
    plot.title(f"r = {r2_score(max, max_model(year))*100:.2f}%", loc = "right", size = 7)
    plot.title("Global Solar Radiation Trend - King's Park")
    plot.xlabel("Year (1992-2021)")
    plot.ylabel("Maximum Solar Radiation (MJ/m^2)")
    plot.show()
    
    
if __name__ == "__main__":
    main()