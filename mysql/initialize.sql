SELECT * 
FROM gsr.daily_kp_gsr_all
WHERE dataCompleteness = 'C';

ALTER TABLE gsr.daily_kp_gsr_all
RENAME COLUMN 'data Completeness' TO data;

SELECT year, avg(value) AS avg, max(value) AS max, min(value) AS min
FROM gsr.daily_kp_gsr_all 
WHERE dataCompleteness = 'C'
GROUP BY year 
ORDER BY year;

select month, year, avg(value) as avg, max(value) as max, min(value) as min, row_number() over(partition by year order by year, month) as id
from gsr.daily_kp_gsr_all
where year>1992 and year<2000
group by month, year
order by year;

/* create test database */
CREATE DATABASE test;

CREATE TABLE test.daily_kp_gsr_all (
    Year SMALLINT NOT NULL,
    Month TINYINT NOT NULL,
    Day TINYINT NOT NULL,
    Value FLOAT NOT NULL,
    dataCompleteness CHAR(1) NOT NULL
);

DROP TABLE test.daily_kp_gsr_all;

SELECT * FROM test.daily_kp_gsr_all;

SHOW VARIABLES LIKE "secure_file_priv";

LOAD DATA INFILE 'C:/Programming/Python/Global-Solar-Radiation-in-King-s-Park/data/daily_KP_GSR_ALL.csv' 
INTO TABLE test.daily_kp_gsr_all 
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 3 ROWS;

SELECT year, avg(value) AS avg, max(value) AS max, min(value) AS min
FROM gsr.daily_kp_gsr_all 
WHERE dataCompleteness = 'C'
GROUP BY year 
ORDER BY year;