import pandas as pd
import os

def generate_files_for_official_eval(results_csv_path="results/comparison_results.csv", output_dir="results"):
    """
    Reads the main experiment results and creates separate .sql prediction files
    for each method, required by the official Spider evaluation script.
    """
    try:
        df = pd.read_csv(results_csv_path)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_csv_path}. Run the main experiment first.")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # The official dev set has a specific order. We need to respect that.
    # The simplest way is to drop duplicates based on question and db_id, 
    # assuming your run_experiment processed them in order.
    unique_questions_df = df.drop_duplicates(subset=['db_id', 'question'])

    methods = df['method'].unique()
    print(f"Found methods: {methods}")

    for method in methods:
        method_df = df[df['method'] == method]
        
        # Merge with the unique questions to ensure the order is correct
        ordered_df = pd.merge(unique_questions_df[['db_id', 'question']], method_df, on=['db_id', 'question'], how='left')
        
        # Sanitize method name for filename
        filename_method = method.replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')
        output_path = os.path.join(output_dir, f"predictions_{filename_method}.sql")
        
        # The official script expects each line to be the predicted SQL query
        ordered_df['predicted_sql'].to_csv(output_path, header=False, index=False, sep='\t')
        
        print(f"Generated prediction file for '{method}' at: {output_path}")

if __name__ == '__main__':
    generate_files_for_official_eval()