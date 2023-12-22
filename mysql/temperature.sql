CREATE DATABASE temperature;
DROP DATABASE test;

/* create temperature database */
CREATE DATABASE temperature;

/* create gsr table */
CREATE TABLE temperature.daily_kp_gsr_all (
    Year SMALLINT NOT NULL,
    Month TINYINT NOT NULL,
    Day TINYINT NOT NULL,
    Value FLOAT NOT NULL,
    dataCompleteness CHAR(1) NOT NULL
);

/* create rh table */
CREATE TABLE temperature.daily_kp_rh_all (
    Year SMALLINT NOT NULL,
    Month TINYINT NOT NULL,
    Day TINYINT NOT NULL,
    Value TINYINT NOT NULL,
    dataCompleteness CHAR(1) NOT NULL
);

/* create sun table */
CREATE TABLE temperature.daily_kp_sun_all (
    Year SMALLINT NOT NULL,
    Month TINYINT NOT NULL,
    Day TINYINT NOT NULL,
    Value FLOAT NOT NULL,
    dataCompleteness CHAR(1) NOT NULL
);

/* create avg_temp table */
CREATE TABLE temperature.clmtemp_kp (
    Year SMALLINT NOT NULL,
    Month TINYINT NOT NULL,
    Day TINYINT NOT NULL,
    Value FLOAT NOT NULL,
    dataCompleteness CHAR(1) NOT NULL
);

DROP TABLE temperature.daily_kp_gsr_all;

SELECT * FROM temperature.daily_kp_gsr_all
WHERE Year = 1992 AND Month = 7;

SHOW VARIABLES LIKE "secure_file_priv";

/* load gsr data */
LOAD DATA INFILE 'C:/Programming/Python/Global-Solar-Radiation-in-King-s-Park/data/daily_KP_GSR_ALL.csv' 
INTO TABLE temperature.daily_kp_gsr_all 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 3 ROWS;

/* load rh data */
LOAD DATA INFILE 'C:/Programming/Python/Global-Solar-Radiation-in-King-s-Park/data/daily_KP_RH_ALL.csv' 
INTO TABLE temperature.daily_kp_rh_all 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 3 ROWS;

/* load sun data */
LOAD DATA INFILE 'C:/Programming/Python/Global-Solar-Radiation-in-King-s-Park/data/daily_KP_SUN_ALL.csv' 
INTO TABLE temperature.daily_kp_sun_all 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 3 ROWS;

/* load avg_temp data */
LOAD DATA INFILE 'C:/Programming/Python/Global-Solar-Radiation-in-King-s-Park/data/CLMTEMP_KP_.csv' 
INTO TABLE temperature.clmtemp_kp
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 3 ROWS;

SELECT year, avg(value) AS avg, max(value) AS max, min(value) AS min
FROM gsr.daily_kp_gsr_all 
WHERE dataCompleteness = 'C'
GROUP BY year 
ORDER BY year;

SELECT * FROM temperature.clmtemp_kp;

SELECT Year, Month, AVG(Value) AS AvgTemp FROM temperature.clmtemp_kp
WHERE dataCompleteness = 'C'
GROUP BY Year, Month
ORDER BY Year, Month;

SELECT * FROM temperature.clmtemp_kp;
SELECT * FROM temperature.daily_kp_gsr_all;
SELECT * FROM temperature.daily_kp_rh_all;
SELECT * FROM temperature.daily_kp_sun_all;

CREATE TABLE temperature.join_result (
    Year SMALLINT NOT NULL,
    Month TINYINT NOT NULL,
    Day TINYINT NOT NULL,
    sun FLOAT NOT NULL,
    rh TINYINT NOT NULL,
    gsr FLOAT NOT NULL,
    avgTemp FLOAT NOT NULL
);

INSERT INTO temperature.join_result
SELECT SUN.Year, SUN.Month, SUN.Day, SUN.Value AS sun, RH.Value AS rh, GSR.Value AS gsr, AVGTEMP.Value AS avgTemp FROM temperature.daily_kp_sun_all SUN
JOIN temperature.daily_kp_rh_all RH
ON SUN.Year = Rh.Year AND SUN.Month = Rh.Month AND SUN.DAY = Rh.DAY AND Rh.dataCompleteness = 'C' AND SUN.dataCompleteness = 'C'
JOIN temperature.daily_kp_gsr_all GSR
ON SUN.Year = GSR.Year AND SUN.Month = GSR.Month AND SUN.DAY = GSR.DAY AND GSR.dataCompleteness = 'C' AND SUN.dataCompleteness = 'C'
JOIN temperature.clmtemp_kp AVGTEMP
ON SUN.Year = AVGTEMP.Year AND SUN.Month = AVGTEMP.Month AND SUN.DAY = AVGTEMP.DAY AND AVGTEMP.dataCompleteness = 'C' AND SUN.dataCompleteness = 'C'
ORDER BY Sun.Year, Sun.Month, Sun.Day;

SELECT sun, rh, gsr, avgTemp FROM temperature.join_result
ORDER BY Year, Month, Day;

SELECT sun, avgTemp FROM temperature.join_result
ORDER BY Year, Month, Day;