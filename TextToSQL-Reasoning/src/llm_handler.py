import openai
import google.generativeai as genai
import anthropic
import time
import logging

from src.config import (
    OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY,
    MAX_RETRIES, INITIAL_RETRY_DELAY
)

# Get the logger that was set up in run_experiment.py
logger = logging.getLogger()

def get_llm_response(prompt: str, provider: str = "openai", model: str = "gpt-4-turbo-preview") -> str:
    """
    Gets a response from the specified LLM provider with robust retry logic.

    Args:
        prompt: The input prompt for the LLM.
        provider: The LLM provider ("openai", "gemini", or "anthropic").
        model: The model name to use.

    Returns:
        The text response from the LLM, or an error message if all retries fail.
    """
    retries = 0
    delay = INITIAL_RETRY_DELAY

    while retries < MAX_RETRIES:
        try:
            if provider == "openai":
                if not OPENAI_API_KEY: raise ValueError("OpenAI API key not found.")
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content.strip()

            elif provider == "gemini":
                if not GEMINI_API_KEY: raise ValueError("Gemini API key not found.")
                genai.configure(api_key=GEMINI_API_KEY)
                gemini_model = genai.GenerativeModel(model)
                response = gemini_model.generate_content(prompt)
                return response.text.strip()

            elif provider == "anthropic":
                if not ANTHROPIC_API_KEY: raise ValueError("Anthropic API key not found.")
                client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                message = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return message.content[0].text.strip()
                
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            # Check if the error is a rate limit error (specific error types might vary by library)
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(f"Rate limit exceeded for {provider}. Retrying in {delay} seconds...")
                time.sleep(delay)
                retries += 1
                delay *= 2  # Exponential backoff
            else:
                # For other errors, log them and stop retrying
                logger.error(f"An unexpected error occurred with the {provider} API: {e}")
                return f"Error: {e}"
    
    logger.error(f"Failed to get response from {provider} after {MAX_RETRIES} retries.")
    return f"Error: Failed after {MAX_RETRIES} retries due to rate limiting."


def parse_sql_from_response(response: str) -> str:
    """
    Parses the SQL query from the LLM's response.
    """
    if response.startswith("Error:"):
        return response # Pass the error message through

    if "```sql" in response:
        response = response.split("```sql")[1].split("```")[0]
    elif "SELECT" in response:
        # Be careful with this, as reasoning text might contain "SELECT"
        # Let's find the last occurrence to be safer
        select_pos = response.rfind("SELECT")
        if select_pos != -1:
            response = response[select_pos:]

    return response.strip().replace("\n", " ").replace(";", "")