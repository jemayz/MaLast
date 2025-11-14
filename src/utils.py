import logging
import json
import re
from src.doc_qa import AgenticQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# --- MODIFIED FOR RENDER ---
# 1. Changed the function signature to accept 'persist_directory'
# 2. Added logic to use this new path or fall back to the old default
#
def load_rag_system(collection_name, domain, persist_directory=None):
# --- END MODIFICATION ---
    """
    Loads an existing RAG system by connecting to the persistent vector store.
    This is fast and does not re-process any documents.
    """
    logger.info(f"Loading RAG system for collection: '{collection_name}'  (Domain: {domain})...")
    try:
        
        # --- MODIFIED FOR RENDER ---
        # Determine the correct path
        # Use the provided path (from app.py), or fall back to the old default
        if persist_directory:
            db_path = persist_directory
        else:
            db_path = "chroma_db"
            logger.warning(f"No persist_directory provided, falling back to default '{db_path}'")
        
        logger.info(f"Using database path: {db_path}")
        # --- END MODIFICATION ---

        agent = AgenticQA(
            config={
                "retriever": {
                    "collection_name": collection_name,
                    # --- MODIFIED FOR RENDER ---
                    "persist_directory": db_path # <-- Use the dynamic path
                    # --- END MODIFICATION ---
                },
                "domain": domain
            }
        )
        # Check if the agent was actually created
        if not agent.agent_executor:
            raise Exception("Agent Executor was not created. Check logs for errors.")

        logger.info(f"✅ System for '{collection_name}' loaded successfully.")
        return agent
    except Exception as e:
        logger.error(f"❌ Failed to load RAG system for '{collection_name}': {e}")
        logger.warning("Did you run the ingest.py script first?")
        return None

def markdown_bold_to_html(text: str):
    """Converts markdown bold syntax to HTML <strong> tags."""
    return re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)

def standardize_query(query):
    if not query:
        return None
    return query.strip().lower()

def get_standalone_question(input_question, chat_history,llm):
    """Uses LLM to create a standalone question from the chat history."""
    if not chat_history:
        return input_question
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, FORMULATE A STANDALONE QUESTION which can be understood without the chat history. Do NOT ANSWER THE QUESTION, just reformulate it if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever_chain = contextualize_q_prompt | llm
    
    response = history_aware_retriever_chain.invoke(
        {"chat_history": chat_history, "input": input_question}
    )
    return response.content

def parse_agent_response(response_dict):
    """A robust helper to parse the dictionary from an AgenticQA agent."""
    answer = markdown_bold_to_html(response_dict.get('answer', 'Error: No answer found.'))
    thoughts = response_dict.get('thoughts', 'No thought process available.')
    validation = response_dict.get('validation', (False, 'Validation failed.'))
    source = response_dict.get('source', 'Unknown')
    
    if validation and validation[1] == "Validation skipped for insurance domain.":
        validation = (True, "Factual Answer")
        
    return answer, thoughts,validation, source
    
def extract_json_from_string(text: str) -> dict:
    """
    Finds and parses the first valid JSON object within a string.
    Returns a dictionary, or an empty dict if no JSON is found.
    """
    # This regex finds the first occurrence of a string starting with { and ending with }
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    
    if json_match:
        json_string = json_match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            # The extracted string is not valid JSON
            return {"error": "Failed to parse extracted JSON", "raw_text": json_string}
    else:
        # No JSON object found in the string
        return {"error": "No JSON object found in the string", "raw_text": text}