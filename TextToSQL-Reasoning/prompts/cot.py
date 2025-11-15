def create_prompt(question: str, schema: str) -> str:
    """Creates a Chain-of-Thought prompt."""
    prompt = f"""{schema}
### Question: {question}
### Reasoning: Let's think step by step. First, I need to find the students enrolled in 'Math'. I can do this by joining the student, enrollment, and course tables. Second, I need to find the students enrolled in 'Physics' using the same joins. Finally, I need to find the students that appear in both lists. I can use an INTERSECT or a subquery with IN for this. I will select the student names for the final result.
### SQL:"""
    return prompt