import sqlite3
import traceback

def execute_sql(cursor, query):
    """Executes a SQL query and returns the results, handling errors."""
    try:
        cursor.execute(query)
        return sorted(cursor.fetchall()) # Sort results to make comparison order-independent
    except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
        # traceback.print_exc() # Uncomment for deep debugging
        return f"Error: {e}"

def evaluate_ex(predicted_sql: str, gold_sql: str, db_path: str) -> bool:
    """
    Performs Execution Accuracy evaluation.
    Connects to the database, runs both queries, and compares the results.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        predicted_results = execute_sql(cursor, predicted_sql)
        gold_results = execute_sql(cursor, gold_sql)

        # If either query resulted in an error, they can't match
        if isinstance(predicted_results, str) or isinstance(gold_results, str):
            return False

        return predicted_results == gold_results

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Example usage (requires a dummy db)
    db_file = "data/spider/database/academic/academic.sqlite"
    gold_q = "SELECT Fname FROM FACULTY WHERE rank = 'Professor'"
    pred_q_correct = "SELECT Fname FROM FACULTY WHERE rank = 'Professor'"
    pred_q_wrong = "SELECT Lname FROM FACULTY WHERE rank = 'Professor'"
    
    print(f"Correct match: {evaluate_ex(pred_q_correct, gold_q, db_file)}")
    print(f"Incorrect match: {evaluate_ex(pred_q_wrong, gold_q, db_file)}")