def create_prompt(question: str, schema: str, use_intercol: bool = False) -> str:
    """Creates a QDecomp or QDecomp+InterCOL prompt."""
    if not use_intercol:
        # QDecomp
        reasoning = """### Decomposed Questions:
1. Which students are enrolled in Math?
2. Which students are enrolled in Physics?
3. Which of these students are in both groups?"""
    else:
        # QDecomp+InterCOL
        reasoning = """### Decomposed Questions with Schema:
1. Find student_name from student by joining with enrollment and course where course_name is 'Math'. SQL tables(columns): student(student_name), course(course_name), enrollment(*).
2. Find student_name from student by joining with enrollment and course where course_name is 'Physics'. SQL tables(columns): student(student_name), course(course_name), enrollment(*).
3. Find the intersection of the students from the previous steps."""

    prompt = f"""{schema}
### Question: {question}
{reasoning}
### SQL:"""
    return prompt