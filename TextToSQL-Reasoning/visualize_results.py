import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RESULTS_PATH

def plot_results():
    """Reads the results CSV and plots a bar chart of the accuracies."""
    try:
        df = pd.read_csv(RESULTS_PATH)
    except FileNotFoundError:
        print(f"Error: Results file not found at {RESULTS_PATH}. Please run the experiment first.")
        return

    # Calculate accuracy per method
    accuracy_summary = df.groupby('method')['is_correct_ex'].mean().reset_index()
    accuracy_summary['accuracy'] = accuracy_summary['is_correct_ex'] * 100
    accuracy_summary = accuracy_summary.sort_values('accuracy', ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='accuracy', y='method', data=accuracy_summary, palette='viridis')
    
    plt.title('Comparison of Prompting Methods for Text-to-SQL', fontsize=16)
    plt.xlabel('Execution Accuracy (%)', fontsize=12)
    plt.ylabel('Prompting Method', fontsize=12)
    
    # Add accuracy labels to the bars
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 1, p.get_y() + p.get_height() / 2,
                 f'{width:.2f}%',
                 va='center')
    
    plt.xlim(0, 100)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'results/accuracy_comparison.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

if __name__ == '__main__':
    plot_results()