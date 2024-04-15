from loguru import logger
from sklearn.model_selection import train_test_split
from polars import DataFrame
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

class dataloader():
    def __init__(self, path:str=os.getcwd(), lag=20):
        self.parent_dir = path
        self.lag = lag
        self.data_dir = os.path.join(self.parent_dir, "data")
        self.AVG_TEMP_DATA = os.path.join(self.data_dir, "CLMTEMP_KP_.csv")
        self.GSR_DATA = os.path.join(self.data_dir, "daily_KP_GSR_ALL.csv")
        self.RH_DATA = os.path.join(self.data_dir, "daily_KP_RH_ALL.csv")
        self.SUN_DATA = os.path.join(self.data_dir, "daily_KP_SUN_ALL.csv")
        self.RF_DATA = os.path.join(self.data_dir, "daily_KP_RF_ALL.csv")
        self.UV_DATA = os.path.join(self.data_dir, "daily_KP_UV_ALL.csv")
        self.WSPD_DATA = os.path.join(self.data_dir, "daily_KP_WSPD_ALL.csv")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")

        self.AVG_TEMP_df = self.load_csv(self.AVG_TEMP_DATA, "AVG_TEMP")
        self.GSR_df = self.load_csv(self.GSR_DATA, "GSR")
        self.RH_df = self.load_csv(self.RH_DATA, "RH")
        self.SUN_df = self.load_csv(self.SUN_DATA, "SUN")
        self.RF_df = self.load_csv(self.RF_DATA, "RF")
        self.UV_df = self.load_csv(self.UV_DATA, "UV")
        self.WSPD_df = self.load_csv(self.WSPD_DATA, "WSPD")
        self.start_date, self.end_date = self.get_overlapped_range(self.AVG_TEMP_df, self.GSR_df, self.RH_df, self.SUN_df, self.RF_df, self.UV_df, self.WSPD_df, data_column='date')

        self.joined_df = self.join_df(self.start_date, self.end_date, self.AVG_TEMP_df, self.GSR_df, self.RH_df, self.SUN_df, self.RF_df, self.UV_df, self.WSPD_df)

        self.features = self.feature_engineering()
        # self.train, self.test = self.split_data(self.features)
        # self.train, self.val = self.split_data(self.train)
        # self.X_train, self.y_train, self.date_train = self.split_xy(self.train)
        # self.X_val, self.y_val, self.date_val = self.split_xy(self.val)
        # self.X_test, self.y_test, self.date_test = self.split_xy(self.test)
        self.train, self.test = self.split_data(self.features)
        self.X_train, self.y_train, self.date_train = self.split_xy(self.train)
        self.X_test, self.y_test, self.date_test = self.split_xy(self.test)
        # self.X_train, self.y_train = self.split_series(self.train, n_past=self.lag, n_future=1)
        # self.X_test, self.y_test = self.split_series(self.test, n_past=self.lag, n_future=1)

    def get_overlapped_range(self, *args:list[DataFrame], data_column):
        start_date = None
        end_date = None

        for df in args:
            # logger.info("="*20)
            if data_column in df.columns:
                date_column = df[data_column]
                if len(date_column) > 0:
                    df_start_date = date_column.min()
                    # logger.info(df_start_date)
                    df_end_date = date_column.max()
                    # logger.info(df_end_date)

                    if start_date is None or df_start_date > start_date:
                        start_date = df_start_date
                    if end_date is None or df_end_date < end_date:
                        end_date = df_end_date

        return start_date, end_date

    def join_df(self, start_date:pl.Date, end_date:pl.Date, *args:list[DataFrame]):
        dfs = []

        for df in args:
            dfs.append(df.filter(pl.col("date").is_between(start_date, end_date)))

        if len(dfs) == 0:
            return None

        joined_df = dfs[0]
        for i in range(1, len(dfs)):
            joined_df = joined_df.join(dfs[i], on="date", how="outer_coalesce")

        # drop columns that end with _DC
        columns_to_keep = [col for col in joined_df.columns if not col.endswith("_DC")]
        joined_df = joined_df.select(columns_to_keep)

        # # interpolate the missing data
        # joined_df = joined_df.interpolate()
        return joined_df

    def fill_na_with_LI(self, df:DataFrame, order:int=3):
        """
        fill the null value with Spline Interpolation
        """
        # Change the interpolation method to 'spline'
        return df.interpolate()

    def load_csv(self, file_path: str, value_name: str) -> DataFrame:
        df = pl.read_csv(file_path, has_header=False, skip_rows=2)
        """
        │ Year     ┆ Month    ┆ Day      ┆ Value    ┆ data Completeness │
        │ 1992     ┆ 7        ┆ 1        ┆ 29.3     ┆ C                 │
        
        *** 沒有數據/unavailable
        # 數據不完整/data incomplete
        C 數據完整/data Complete
        
        """

        # turn the first row to column names
        df = df.rename(df.head(1).to_dicts().pop())
        # drop the first row
        df = df[1:]
        # concatenate the year, month, day to date
        df = df.with_columns(pl.date(df['Year'], df['Month'], df['Day']).alias('date'))
        # drop columns year, month, day
        df = df.drop(['Year', 'Month', 'Day'])
        # drop the row that is null in column value and data completeness
        df = df.filter(~pl.all_horizontal(pl.all().is_null()))
        # replace the value *** in value column to null
        df = df.with_columns(pl.col(pl.Utf8).replace("***", None))
        # turn the value column to float
        df = df.with_columns(pl.col(df.columns[0]).cast(pl.Float32, strict=False).alias(df.columns[0]))
        # replace the value column name to the value_name
        df = df.rename({df.columns[0]: value_name, df.columns[1]: f"{value_name}_DC"}) 
        # df = self.fill_na_with_LI(df)
        logger.info(df)
        # logger.info the null value count in each column
        logger.info(df.null_count())
        # pl.from_pandas(df.to_pandas().set_index("date"))
        return df

    def plot_df(self, df: pl.DataFrame, title: str):
        """
        Plots the first column of the DataFrame against the 'date' column using a scatter plot.

        Args:
            df: The DataFrame to plot.
            title: The title of the plot.
        """
        # Select the 'date' column and the first remaining column (assuming it's the data)
        date_col = "date"
        data_col = df.columns[0]  # Assuming the first column is 'date'
        plot_data = df[[date_col, data_col]]

        # Get minimum and maximum values for y-axis
        # logger.info(df.head(5))
        min_value = df[data_col].min()
        max_value = df[data_col].max()

        logger.info(f"Min value: {min_value}, Max value: {max_value}")
        ticks = np.arange(int(min_value)-2, int(max_value)+2, step=2)
        logger.info(ticks)
        # Create the scatter plot
        sns.scatterplot(x=date_col, y=data_col, data=plot_data)
        plt.yticks(ticks=ticks)

        plt.title(title)
        plt.show()
        return

    def remove_outliers(self, df:DataFrame, column:str, threshold:float=5):
        """
        Remove the outliers in the column of the DataFrame
        """
        logger.debug(df.head())
        # df = df[column]
        # remove the null value
        mean = df.mean()[column][0]
        std = df.std()[column][0]
        mask = (df[column] < mean + threshold * std) & (df[column] > mean - threshold * std)
        df = df.filter(mask)
        return df

    def weather_feature(self, df:DataFrame):
        """
        Add weather features to the DataFrame
        if Sun value is 0, then it is 0, otherwise 1
        """
        df = df.with_columns(pl.col("SUN").map_elements(lambda x: 0 if x == 0 else 1, return_dtype=pl.Int32).alias("weather"))        
        return df

    def seasonal_decompose(self, df:DataFrame, model:str='additive', period:int=365):
        """
        Seasonal decomposition using moving averages
        """
        df = df.with_columns(pl.col("date").dt.weekday().alias("dayofweek"))
        df = df.with_columns(pl.col("date").dt.month().alias("Month"))
        df = df.with_columns(pl.col("date").dt.year().alias("year"))
        df = df.with_columns(pl.col("date").dt.day().alias("dayofmonth"))
        df = df.with_columns(pl.col("date").dt.week().alias("weekofyear"))
        df = df.with_columns(pl.col("date").dt.ordinal_day().alias("dayofyear"))
        df = df.with_columns(pl.col("date").dt.quarter().alias("quarter"))
        return df

    def plot_avg_temp(self, df:DataFrame, column:str="AVG_TEMP"):
        """
        Plot the average temperature
        """
        fig, ax = plt.subplots(figsize=(15, 5),)
        sns.boxplot(data=df, x='dayofyear', y=column)
        ax.set_title(f'{column} by Day in a Year')

        # Set the x-axis tick labels
        x_ticks = ax.get_xticks()
        x_labels = ax.get_xticklabels()
        new_x_labels = [label if i % 29 == 0 else '' for i, label in enumerate(x_labels)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(new_x_labels)

        plt.show()
        return

    # transform a time series dataset into a supervised learning dataset
    def series_to_supervised(self, df:DataFrame, time_series_key:str, n_in=0, n_out=0):
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            df = df.with_columns(pl.col(time_series_key).shift(i).alias(time_series_key+'_t-'+str(i)))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(1, n_out+1):
            df = df.with_columns(pl.col(time_series_key).shift(-i).alias(time_series_key+'_t+'+str(i)))

        return df[n_in:]

    def split_series(self, df:DataFrame, n_past=0, n_future=1):
        #
        # n_past ==> no of past observations
        #
        # n_future ==> no of future observations
        #
        X, y = list(), list()
        for window_start in range(len(df)):
            past_end = window_start + n_past
            future_end = past_end + n_future
            if future_end > len(df):
                break
            # slicing the past and future parts of the window
            past = df.slice(window_start, n_past).drop(["date"])
            future = df.slice(
                past_end, n_future
            ).drop(
                [
                    "date",
                    "dayofweek",
                    "Month",
                    "year",
                    "dayofmonth",
                    "weekofyear",
                    "dayofyear",
                    "quarter",
                ]
            )["AVG_TEMP"]
            X.append(past)
            y.append(future)
        return X, y

    def feature_engineering(self):

        # remove the outliers
        logger.info("RF")
        rf_threshold = 6
        rf_df = self.remove_outliers(self.RF_df, 'RF', 6), f'after remove outliers RF, threshold={rf_threshold}_sd'
        logger.info("WSPD")
        wspd_threshold = 4
        wspd_df = self.remove_outliers(self.WSPD_df, 'WSPD', 4), f'after remove outliers WSPD, threshold={wspd_threshold}_sd'
        rf_removed_count = len(self.RF_df) - len(rf_df[0])
        wspd_removed_count = len(self.WSPD_df) - len(wspd_df[0])
        logger.success(f"Removed {rf_removed_count} outliers from RF")
        logger.success(f"Removed {wspd_removed_count} outliers from WSPD")

        joined_df = self.join_df(self.start_date, self.end_date, self.AVG_TEMP_df, self.GSR_df, self.SUN_df, self.RH_df, self.UV_df, rf_df[0], wspd_df[0])
        joined_df = pl.DataFrame(joined_df)

        # transform timestamp to datetime
        from datetime import datetime
        time_df = pl.datetime_range(
            datetime(self.start_date.year, self.start_date.month, self.start_date.day), 
            datetime(self.end_date.year, self.end_date.month, self.end_date.day),
            "1d", eager=True
        ).alias("datetime")
        
        joined_df = joined_df.with_columns([time_df.alias("date")])
        logger.error(joined_df)
        # perform linear interpolation
        joined_df = self.fill_na_with_LI(joined_df)

        # add weather feature
        joined_df = self.weather_feature(joined_df)

        # seasonal decomposition
        joined_df = self.seasonal_decompose(joined_df)

        logger.info(joined_df)
        logger.info(joined_df.null_count())
        # shift the average temperature to the next day
        # joined_df = self.series_to_supervised(df=joined_df, time_series_key="AVG_TEMP", n_in=self.lag, n_out=0) # past 20 days

        # logger.info(joined_df.null_count())

        # Turn all columns with same data type to float
        for col in joined_df.columns:
            if col == "date":
                continue
            if not joined_df[col].dtype == pl.Float32:
                joined_df = joined_df.with_columns(pl.col(col).cast(pl.Float32, strict=False).alias(col))
        logger.info(joined_df)

        # joined_df = self.split_series(joined_df, n_past=self.lag, n_future=1)
        # logger.info(joined_df.shape)

        return joined_df

    def split_data(self, df:DataFrame):
        # split the data to train and test
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
        return train_df, test_df

    def split_xy(self, df:DataFrame):
        x_df = df.drop(["AVG_TEMP", "date"])
        y_df = df["AVG_TEMP"]
        date = df["date"]
        return x_df, y_df, date

if __name__ == "__main__":
    data = dataloader()
    # data.plot_df(data.AVG_TEMP_df[:730], "AVG_TEMP")
    # data.plot_df(data.GSR_df, "GSR")
    # data.plot_df(data.RH_df, "RH")
    # data.plot_df(data.SUN_df, "SUN")
    # data.plot_df(data.UV_df, "UV")
    
    # data.plot_df(data.RF_df, "RF") # eliminate the extreme data
    # data.plot_df(data.WSPD_df, "WSPD") # eliminate the extreme data
    
    # After remove the outliers

    # logger.info(data.start_date, data.end_date)
    logger.success(data.features)
    logger.success(data.features.null_count())
    
    # data.plot_avg_temp(data.features, "AVG_TEMP")
    # data.plot_avg_temp(data.features, "GSR")
    # data.plot_avg_temp(data.features, "SUN")
    # data.plot_avg_temp(data.features, "RH")
    # data.plot_avg_temp(data.features, "UV")
    # data.plot_avg_temp(data.features, "RF")
    # data.plot_avg_temp(data.features, "WSPD")
    # data.plot_avg_temp(data.features, "weather")
    
    logger.success(data.X_train)
    logger.success(data.y_train)
    logger.success(data.X_test)
    logger.success(data.y_test)
