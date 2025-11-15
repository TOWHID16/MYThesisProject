import time

import pandas as pd
import os
import logging
from datetime import datetime
from tqdm import tqdm
from src.config import REQUEST_DELAY 
from src.config import NUM_SAMPLES, RESULTS_PATH, DEFAULT_PROVIDER, DEFAULT_MODEL
from src.dataset_loader import load_spider_dataset, get_schema_from_tables_json
from src.llm_handler import get_llm_response, parse_sql_from_response
from src.evaluator import evaluate_ex

# Import prompt creation functions
from prompts import standard, cot, qdecomp, scot

def setup_logging():
    """Sets up logging to both console and a file with a unique timestamp."""
    # Create a unique log filename with a timestamp
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"experiment_{timestamp}.log")

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a handler to write to the log file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a handler to print to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def run_experiment():
    logger = setup_logging()
    logger.info("--- Starting Text-to-SQL Experiment ---")
    
    # 1. Load Dataset
    dataset = load_spider_dataset()
    if not dataset:
        logger.error("Dataset could not be loaded. Exiting experiment.")
        return
    
    # Use a subset for quick testing
    if NUM_SAMPLES > 0:
        dataset = dataset[:NUM_SAMPLES]
    logger.info(f"Running experiment on {len(dataset)} samples.")
    
    results = []
    
    # 2. Iterate through dataset
    for item in tqdm(dataset, desc="Processing Questions"):
        question = item['question']
        gold_query = item['query']
        db_id = item['db_id']
        db_path = f"data/spider/database/{db_id}/{db_id}.sqlite"
        
        if not os.path.exists(db_path):
            logger.warning(f"Database path not found: {db_path}. Skipping item for DB ID: {db_id}.")
            continue
            
        schema = get_schema_from_tables_json(db_id)

        # 3. Define methods to test
        methods = {
            "Standard": standard.create_prompt(question, schema),
            "CoT": cot.create_prompt(question, schema),
            "QDecomp": qdecomp.create_prompt(question, schema, use_intercol=False),
            "QDecomp+InterCOL": qdecomp.create_prompt(question, schema, use_intercol=True),
            "SCoT (Ours)": scot.create_prompt(question, schema)
        }
        
        # 4. Run each method
        for method_name, prompt in methods.items():
            logger.info(f"--- Running Method: {method_name} for DB: {db_id} ---")
            
            # 5. Get LLM response
            raw_response = get_llm_response(prompt, provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL)
            predicted_sql = parse_sql_from_response(raw_response)
            
            logger.info(f"Question: {question}")
            logger.info(f"Predicted SQL: {predicted_sql}")

            # 6. Evaluate
            is_correct = evaluate_ex(predicted_sql, gold_query, db_path)
            
            # 7. Log result
            results.append({
                "db_id": db_id,
                "question": question,
                "gold_query": gold_query,
                "method": method_name,
                "prompt": prompt,
                "raw_response": raw_response,
                "predicted_sql": predicted_sql,
                "is_correct_ex": is_correct
            })

    # 8. Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"--- Experiment Finished. Results saved to {RESULTS_PATH} ---")
    
    # 9. Print summary
    summary = results_df.groupby('method')['is_correct_ex'].mean().reset_index()
    summary['accuracy'] = summary['is_correct_ex'] * 100
    
    # Convert the summary DataFrame to a string to log it nicely
    summary_string = summary[['method', 'accuracy']].round(2).to_string()
    logger.info(f"\n--- Accuracy Summary ---\n{summary_string}")

if __name__ == '__main__':
    run_experiment()