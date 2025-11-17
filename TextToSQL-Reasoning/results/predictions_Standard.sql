SELECT count(*) FROM singers
SELECT count(*) FROM singers
SELECT Name, Country, Age FROM singer ORDER BY Age DESC
SELECT name, country, age FROM singers ORDER BY age DESC
SELECT AVG(T1.age), MIN(T1.age), MAX(T1.age) FROM singers AS T1 INNER JOIN countries AS T2 ON T1.country_id = T2.country_id WHERE T2.country_name = 'France'
SELECT AVG(S.age) AS average_age, MIN(S.age) AS minimum_age, MAX(S.age) AS maximum_age FROM singers AS S JOIN countries AS C ON S.country_id = C.country_id WHERE C.country_name = 'France'
SELECT T1.song_name, T1.release_year FROM songs AS T1 INNER JOIN singers AS T2 ON T1.singer_id = T2.singer_id ORDER BY T2.age LIMIT 1
SELECT T1.song_name, T1.release_year FROM songs AS T1 INNER JOIN singers AS T2 ON T1.singer_id = T2.singer_id WHERE T2.age = ( SELECT MIN(age) FROM singers )
SELECT DISTINCT Country FROM singers WHERE Age > 20
SELECT DISTINCT country FROM singers WHERE age > 20
SELECT T1.name, COUNT(T2.singer_id) FROM countries AS T1 LEFT JOIN singers AS T2 ON T1.country_id = T2.country_id GROUP BY T1.name
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
