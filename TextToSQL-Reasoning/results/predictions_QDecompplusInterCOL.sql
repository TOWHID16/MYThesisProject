SELECT count(*) FROM singer
SELECT T1.student_name FROM student AS T1 JOIN enrollment AS T2 ON T1.student_id = T2.student_id JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.course_name = 'Math' INTERSECT SELECT T1.student_name FROM student AS T1 JOIN enrollment AS T2 ON T1.student_id = T2.student_id JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.course_name = 'Physics'
SELECT name, country, age FROM singer ORDER BY age DESC
SELECT Name, Country, Age FROM singer ORDER BY Age DESC
SELECT AVG(age), MIN(age), MAX(age) FROM singers WHERE nationality = 'France'
SELECT AVG(age) AS average_age, MIN(age) AS minimum_age, MAX(age) AS maximum_age FROM singers WHERE country = 'France'
SELECT T1.song_name, T1.release_year FROM Songs AS T1 JOIN Singers AS T2 ON T1.singer_id = T2.singer_id WHERE T2.birth_year = ( SELECT MAX(birth_year) FROM Singers )
SELECT T2.song_name, T2.release_year FROM singers AS T1 INNER JOIN songs AS T2 ON T1.singer_id = T2.singer_id WHERE T1.age = ( SELECT MIN(age) FROM singers )
SELECT DISTINCT country FROM singers WHERE age > 20
SELECT DISTINCT T2.country_name FROM singers AS T1 INNER JOIN countries AS T2 ON T1.country_id = T2.country_id WHERE T1.age > 20
SELECT T1.cName, COUNT(T2.sID) FROM country AS T1 LEFT JOIN singer AS T2 ON T1.cName = T2.country GROUP BY T1.cName
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
Error: Failed after 5 retries due to rate limiting.
