SELECT COUNT(*) FROM singer
SELECT COUNT(*) FROM singers
SELECT Name, Country, Age FROM singer ORDER BY Age DESC
SELECT Name, Country, Age FROM singer ORDER BY Age DESC
SELECT AVG(age), MIN(age), MAX(age) FROM singer WHERE country = 'France'
SELECT AVG(age) AS average_age, MIN(age) AS minimum_age, MAX(age) AS maximum_age FROM singers WHERE nationality = 'French'
SELECT T2.Title, T2.ReleaseYear FROM Singers AS T1 INNER JOIN Songs AS T2 ON T1.SingerID = T2.SingerID WHERE T1.Age = ( SELECT MIN(Age) FROM Singers )
SELECT T2.title, T2.release_year FROM singers AS T1 INNER JOIN songs AS T2 ON T1.singer_id = T2.singer_id WHERE T1.age = ( SELECT MIN(age) FROM singers )
SELECT DISTINCT sCountry FROM singer WHERE sAge  >  20
SELECT DISTINCT T2.country_name FROM singers AS T1 INNER JOIN countries AS T2 ON T1.country_id = T2.country_id WHERE T1.age > 20
SELECT T1.country_name, COUNT(T2.singer_id) AS num_singers FROM countries AS T1 LEFT JOIN singers AS T2 ON T1.country_id = T2.country_id GROUP BY T1.country_name
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
SELECT location, name FROM stadiums WHERE capacity BETWEEN 5000 AND 10000
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
