import mysql.connector
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import numpy as np
from scipy import stats

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
        "select month, avg(value) as avg, max(value) as max, min(value) as min, row_number() over(partition by year order by year, month) as id\
        from gsr.daily_kp_gsr_all\
        where year>=1993 and year<=2021\
        group by month, year\
        order by year"
    )
    result_2020 = mycursor.fetchall()
    #turn data to x,y
    month = [] #0
    avg = [] #1
    tol_avg = []
    max = [] #2
    min = [] #3
    id = [] #4
    index = 0

    tol_month = list(range(1,13))
    print(tol_month)
    print(result_2020)
    for data in result_2020:
        month.append(data[0])
        avg.append(data[1])
        tol_avg.append(data[1])
        max.append(data[2])
        min.append(data[3])

        if data[0] == 12: #if month returns to jan then set to a year and plot it
            print(data)
            month = np.array(month)
            avg = np.array(avg)
            max = np.array(max)
            min = np.array(min)
            plot.plot(month, avg) # same year same colour
            index += 1
            month = [] #0
            avg = [] #1
            max = [] #2
            min = [] #3
    """
    slope_avg, intercept_avg, r_avg, p_avg, std_err_avg = stats.linregress(tol_month, tol_avg) #pass list
    gsr_model_avg = list(map(lambda x: slope_avg * x + intercept_avg, tol_month))
    """

    """
    slope_max, intercept_max, r_max, p_max, std_err_max = stats.linregress(year, max) #pass list
    gsr_model_max = list(map(lambda x: slope_max * x + intercept_max, year))
    """
    
    #plot 1
    #plot.plot(tol_month, gsr_model_avg, color="green")
    plot.title("Global Solar Radiation Trend - King's Park")
    #plot.title(f"r = {r_avg*100:.2f}%", loc = "right", size = 7)
    plot.xlabel("Month (1-12)")
    plot.ylabel("Average Solar Radiation between 1992-2021 (MJ/m^2)")
    '''
    #plot 2
    plot.subplot(1,2,2)
    plot.scatter(year, max)
    plot.plot(year, gsr_model_max, color="green")
    plot.title("Global Solar Radiation Trend - King's Park")
    plot.title(f"r = {r_max*100:.2f}%", loc = "right", size = 7)
    plot.xlabel("Year (1992-2021)")
    plot.ylabel("January Maximum Solar Radiation (MJ/m^2)")
    '''
    plot.show()
    
    
if __name__ == "__main__":
    main()