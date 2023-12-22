import mysql.connector
import matplotlib.pyplot as plot
import numpy as np
from scipy import stats

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
        "select year, avg(value) as avg, max(value) as max, min(value) as min\
        from gsr.daily_kp_gsr_all \
        group by year \
        order by year"
    )
    result_2020 = mycursor.fetchall()
    #print(result_2020)
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

    slope_avg, intercept_avg, r_avg, p_avg, std_err_avg = stats.linregress(year, avg) #pass list
    gsr_model_avg = list(map(lambda x: slope_avg * x + intercept_avg, year))
    print(f"r_avg = {r_avg*100:.2f}%")
    print(f"p_avg = {p_avg*100:.2f}%") #p-value is under 0.05 / 5% indicate that the linear regression is true

    slope_max, intercept_max, r_max, p_max, std_err_max = stats.linregress(year, max) #pass list
    gsr_model_max = list(map(lambda x: slope_max * x + intercept_max, year))
    print(f"r_max = {r_max*100:.2f}%")
    print(f"p_max = {p_max*100:.2f}%")

    year = np.array(year)
    avg = np.array(avg)
    max = np.array(max)
    min = np.array(min)

    #plot 1
    plot.subplot(1,2,1)
    plot.scatter(year, avg)
    plot.plot(year, gsr_model_avg, color="green")
    plot.title("Global Solar Radiation Trend - King's Park")
    plot.title(f"r = {r_avg*100:.2f}%", loc = "right", size = 7)
    plot.xlabel("Year (1992-2021)")
    plot.ylabel("Average Solar Radiation (MJ/m^2)")
    #plot 2
    plot.subplot(1,2,2)
    plot.scatter(year, max)
    plot.plot(year, gsr_model_max, color="green")
    plot.title("Global Solar Radiation Trend - King's Park")
    plot.title(f"r = {r_max*100:.2f}%", loc = "right", size = 7)
    plot.xlabel("Year (1992-2021)")
    plot.ylabel("Maximum Solar Radiation (MJ/m^2)")
    plot.show()
    
    
if __name__ == "__main__":
    main()