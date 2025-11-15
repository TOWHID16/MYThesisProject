import json
from datasets import load_dataset

def load_spider_dataset(split='validation'):
    """
    Loads the Spider dataset using the Hugging Face datasets library.
    'validation' corresponds to the dev set.
    """
    try:
        dataset = load_dataset("spider", split=split)
        print(f"Successfully loaded Spider '{split}' split.")
        return list(dataset)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure you have an internet connection and the dataset is available.")
        return []

def get_schema_from_tables_json(db_id, tables_file_path="data/spider/tables.json"):
    """
    Reads the schema from the tables.json file for a given db_id.
    Formats it into a string similar to the 'API Docs' format from the paper.
    """
    with open(tables_file_path, 'r') as f:
        tables_data = json.load(f)

    db_schema = next((db for db in tables_data if db['db_id'] == db_id), None)

    if not db_schema:
        return "Schema not found."

    schema_str = "### SQLite SQL tables, with their properties:\n#\n"
    for table in db_schema['table_names_original']:
        table_info = next((t for t in db_schema['column_names_original'] if t[0] == -1 and t[1].lower() == table.lower()), None)
        if not table_info: continue

        columns = [col[1] for col in db_schema['column_names_original'] if col[0] == db_schema['table_names_original'].index(table)]
        schema_str += f"# {table} ({', '.join(columns)})\n"
    
    return schema_str

if __name__ == '__main__':
    # Example usage:
    spider_dev_data = load_spider_dataset()
    if spider_dev_data:
        # Get the first item in the dev set
        sample = spider_dev_data[0]
        db_id = sample['db_id']
        question = sample['question']
        query = sample['query']
        
        print(f"Database ID: {db_id}")
        print(f"Question: {question}")
        print(f"Gold SQL Query: {query}")
        
        # Get and print the schema
        schema = get_schema_from_tables_json(db_id)
        print("\n--- Database Schema ---")
        print(schema)