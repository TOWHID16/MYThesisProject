SELECT count(*) FROM singer
SELECT COUNT(*) FROM singer
SELECT name, country, age FROM singer ORDER BY age DESC
SELECT Name, Country, Age FROM singer ORDER BY Age DESC
SELECT AVG(age), MIN(age), MAX(age) FROM singer WHERE country = 'France'
SELECT AVG(age), MIN(age), MAX(age) FROM singer WHERE country = 'France'
SELECT T1.song_name, T1.release_year FROM songs AS T1 INNER JOIN singers AS T2 ON T1.singer_id = T2.singer_id WHERE T2.birth_year = ( SELECT MAX(birth_year) FROM singers )
SELECT T1.name, T1.release_year FROM Songs AS T1 INNER JOIN Singers AS T2 ON T1.singer_id = T2.singer_id WHERE T2.age = ( SELECT MIN(age) FROM Singers )
SELECT DISTINCT country FROM singer WHERE age > 20
SELECT DISTINCT Country FROM singer WHERE Age  >  20
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
