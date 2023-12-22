from loguru import logger
from sklearn.model_selection import train_test_split
import pandas as pd

def DataJoining(AVG_TEMP_df: pd.DataFrame, GSR_df: pd.DataFrame, RH_df: pd.DataFrame, SUN_df: pd.DataFrame, RF_df: pd.DataFrame, UV_df: pd.DataFrame, WSPD_df: pd.DataFrame):
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
    print("=" * 80)
    print(f"correlation matrix: \n{corr}")
    print("=" * 80)
    # Seperate x and y df
    
    target_variable = 'AVG_TEMP'
    drop_list = [target_variable]
    # if corr is smaller than 0 then append to drop_list
    for column in corr.columns:
        if corr[column]['AVG_TEMP'] < 0:
            drop_list.append(column)
    
    x_df = df.drop(drop_list, axis=1)
    y_df = df[target_variable]
    # logger.info(x_df)
    # logger.info(y_df)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test