def create_prompt(question: str, schema: str) -> str:
    """Creates a standard text-to-SQL prompt."""
    prompt = f"""{schema}
### Question: {question}
### SQL:"""
    return prompt

# Example Output for "Math and Physics" question:
# ### SQLite SQL tables, with their properties:
# #
# # student (student_id, student_name)
# # course (course_id, course_name)
# # enrollment (student_id, course_id)
# ### Question: List the names of students who are enrolled in both Math and Physics courses.
# ### SQL: