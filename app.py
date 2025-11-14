from flask import Flask, request, render_template, session, url_for, redirect, jsonify
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import logging
import re
import traceback
import base64
import shutil
import zipfile
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from flask_session import Session

# --- MODIFIED FOR PYTHONANYWHERE ---
# Gets the directory where app.py is located, e.g., /home/YourUsername/YourProject
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Define all persistent paths relative to this project directory
DB_DIR = os.path.join(PROJECT_DIR, "chroma_db")
UPLOAD_DIR = os.path.join(PROJECT_DIR, "Uploads")
SESSION_DIR = os.path.join(PROJECT_DIR, "flask_session")

# Create these directories if they don't exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
# --- END MODIFICATION ---

# --- Core Application Imports ---
# Make sure you have an empty __init__.py file in your 'src' folder

from src.medical_swarm import run_medical_swarm
from src.utils import load_rag_system, standardize_query, get_standalone_question, parse_agent_response, markdown_bold_to_html
from langchain_google_genai import ChatGoogleGenerativeAI

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- 1. DATABASE SETUP FUNCTION (For Deployment) ---
def setup_database():
    """Downloads and unzips the ChromaDB folder from Hugging Face Datasets."""
    
    # --- !!! IMPORTANT !!! ---
    # YOU MUST CHANGE THIS to your Hugging Face Dataset repo ID
    # For example: "your_username/your_database_repo_name"
    DATASET_REPO_ID = "WanIrfan/atlast-db" 
    # -------------------------

    ZIP_FILENAME = "chroma_db.zip"
    
    # --- MODIFIED FOR PYTHONANYWHERE ---
    global DB_DIR, PROJECT_DIR # Use the global variables we defined
    # --- END MODIFICATION ---

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        # --- MODIFIED FOR PYTHONANYWHERE ---
        logger.info(f"✅ Database directory already exists at {DB_DIR}. Skipping download.")
        # --- END MODIFICATION ---
        return

    logger.info(f"📥 Downloading database from HF Hub: {DATASET_REPO_ID}")
    try:
        zip_path = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=ZIP_FILENAME,
            repo_type="dataset",
            # You might need to add your HF token to secrets if the dataset is private
            # token=os.getenv("HF_TOKEN") 
        )
        
        # --- MODIFIED FOR PYTHONANYWHERE ---
        logger.info(f"📦 Unzipping database to {PROJECT_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(PROJECT_DIR) # Extracts to /home/YourUsername/YourProject
        # --- END MODIFICATION ---
            
        logger.info("✅ Database setup complete!")
        
        # Clean up the downloaded zip file to save space
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR setting up database: {e}", exc_info=True)
        # This will likely cause the RAG system to fail loading, which is expected
        # if the database isn't available.

# --- RUN DATABASE SETUP *BEFORE* INITIALIZING THE APP ---
setup_database()


# --- STANDARD FLASK APP INITIALIZATION ---
app = Flask(__name__)
app.secret_key = "a_really_strong_static_secret_key_12345" 
# ✅ Configure cookie-based sessions with larger payload
app.config['SESSION_COOKIE_SECURE'] = False  # Set True if using HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# --- CONFIGURE SERVER-SIDE SESSIONS (MODIFIED FOR PYTHONANYWHERE) ---
# This will use the persistent disk path /home/YourUsername/YourProject/flask_session
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = SESSION_DIR  # Use the dynamic path
Session(app)

# --- SESSION DEBUG BLOCK ---
logger.info(f"--- SESSION DEBUG ---")
logger.info(f"SESSION_DIR set to: {SESSION_DIR}")
try:
    if os.access(SESSION_DIR, os.W_OK):
        logger.info(f"✅ SUCCESS: Session directory {SESSION_DIR} is WRITABLE.")
    else:
        logger.error(f"❌ FAILED: Session directory {SESSION_DIR} is NOT WRITABLE.")
except Exception as e:
    logger.error(f"❌ FAILED: Error checking/creating session directory {SESSION_DIR}: {e}")
logger.info(f"--- END SESSION DEBUG ---")
# --- END MODIFICATION ---


google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logger.warning("⚠️ GOOGLE_API_KEY not found in environment variables. LLM calls will fail.")
else:
    logger.info("GOOGLE_API_KEY loaded successfully.")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.05, google_api_key=google_api_key)

# --- LOAD RAG SYSTEMS (AFTER DB SETUP) ---
logger.info("🌟 Starting Multi-Domain AI Assistant...")
try:
    # --- MODIFIED FOR PYTHONANYWHERE ---
    # This code is now correct, as DB_DIR points to the right path
    # ⚠️ IMPORTANT: You must update your 'load_rag_system' function in 'src/utils.py'
    # to accept 'persist_directory' as an argument
    # ---
    rag_systems = {
        'medical': load_rag_system(
            collection_name="medical_csv_Agentic_retrieval", 
            domain="medical",
            persist_directory=DB_DIR
        ),
        'islamic': load_rag_system(
            collection_name="islamic_texts_Agentic_retrieval", 
            domain="islamic",
            persist_directory=DB_DIR
        ),
        'insurance': load_rag_system(
            collection_name="etiqa_Agentic_retrieval", 
            domain="insurance",
            persist_directory=DB_DIR
        )
    }
    # --- END MODIFICATION ---
except Exception as e:
    logger.error(f"❌ FAILED to load RAG systems. Check database path and permissions. Error: {e}", exc_info=True)
    rag_systems = {'medical': None, 'islamic': None, 'insurance': None}

# Store systems and LLM on the app for blueprints
app.rag_systems = rag_systems
app.llm = llm


# Check initialization status
logger.info("\n📊 SYSTEM STATUS:")
for domain, system in rag_systems.items():
    status = "✅ Ready" if system else "❌ Failed (DB missing?)" 
    logger.info(f"   {domain}: {status}")

def hydrate_history(raw_history_list: list) -> list:
    """Converts a list of dicts from session back into LangChain Message objects."""
    history = []
    if not raw_history_list:
        return history
    for item in raw_history_list:
        if item.get('type') == 'human':
            history.append(HumanMessage(content=item.get('content', '')))
        elif item.get('type') == 'ai':
            history.append(AIMessage(content=item.get('content', '')))
    return history

def dehydrate_history(history_messages: list) -> list:
    """Converts LangChain Message objects into a JSON-serializable list of dicts."""
    raw_list = []
    for msg in history_messages:
        if isinstance(msg, HumanMessage):
            raw_list.append({'type': 'human', 'content': msg.content})
        elif isinstance(msg, AIMessage):
            raw_list.append({'type': 'ai', 'content': msg.content})
    return raw_list

# --- FLASK ROUTES ---

@app.route("/")
def homePage():
    # Clear all session history when visiting the home page
    session.pop('medical_history', None)
    session.pop('islamic_history', None)
    session.pop('insurance_history', None)
    session.pop('current_medical_document', None)
    return render_template("homePage.html")


@app.route("/medical", methods=["GET", "POST"])
def medical_page():
    if request.method == "GET":
        # ✅ USE .get() instead of .pop() - don't remove it yet
        latest_response = session.get('latest_medical_response', {})
        
        answer = latest_response.get('answer', "")
        thoughts = latest_response.get('thoughts', "")
        validation = latest_response.get('validation', "")
        source = latest_response.get('source', "")

        # ✅ NOW clear it after reading (for next request)
        if latest_response:
            session.pop('latest_medical_response', None)
            session.modified = True
        
        # Load history
        raw_history_list = session.get('medical_history', [])
        history = hydrate_history(raw_history_list)
        
        return render_template("medical_page.html", 
                               history=history,  # ✅ Pass hydrated history
                               answer=answer,
                               thoughts=thoughts,
                               validation=validation,
                               source=source)
    
    # POST Request
    answer, thoughts, validation, source = "", "", "", ""
    raw_history_list = session.get('medical_history', [])
    history_for_agent = hydrate_history(raw_history_list)
    current_medical_document = session.get('current_medical_document', "")
    
    try:
        query = standardize_query(request.form.get("query", ""))
        has_image = 'image' in request.files and request.files['image'].filename
        has_document = 'document' in request.files and request.files['document'].filename
        has_query = request.form.get("query") or request.form.get("question", "")
            
        logger.info(f"POST request received: has_image={has_image}, has_document={has_document}, has_query={has_query}")
            
        if has_document:
            logger.info("Processing Scenario 3: Query + Document with Medical Swarm")
            file = request.files['document']
            try:
                document_text = file.read().decode("utf-8")
                session['current_medical_document'] = document_text
                current_medical_document = document_text
            except UnicodeDecodeError:
                answer = "Error: Could not decode the uploaded document. Please ensure it is a valid text or PDF file."
                logger.error("Scenario 3: Document decode error")
                thoughts = traceback.format_exc()
                  
            swarm_answer = run_medical_swarm(current_medical_document, query)
            answer = markdown_bold_to_html(swarm_answer)
                
            thoughts = "Swarm analysis complete. The process is orchestrated and does not use the ReAct thought process. You can now ask follow-up questions."
            source = "Medical Swarm"
            validation = "Swarm output generated."
            
            history_for_agent.append(HumanMessage(content=f"[Document Uploaded] Query: '{query}'"))
            history_for_agent.append(AIMessage(content=answer))
            
        elif has_image:
            logger.info("Processing Multimodal RAG: Query + Image")
            file = request.files['image']
            
            # --- MODIFIED FOR PYTHONANYWHERE ---
            global UPLOAD_DIR
            image_path = os.path.join(UPLOAD_DIR, file.filename)
            # --- END MODIFICATION ---
            
            try:
                file.save(image_path)
                file.close()
            
                with open(image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode("utf-8")
                
                vision_prompt = f"Analyze this image and identify the main subject in a single, concise sentence. The user's query is: '{query}'"
                message = HumanMessage(content=[
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_data}"}
                ])
                vision_response = llm.invoke([message])
                visual_prediction = vision_response.content
                logger.info(f"Vision Prediction: {visual_prediction}")

                enhanced_query = (
                    f'User Query: "{query}" '
                    f'Context from an image provided by the LLM: "{visual_prediction}" '
                    'Based on the user\'s query and the context from LLM, provide a comprehensive answer.'
                )
                logger.info(f"Enhanced query: {enhanced_query}")
            
                agent = rag_systems['medical']
                if not agent: 
                    raise Exception("Medical RAG system is not loaded.")
                
                response_dict = agent.answer(enhanced_query, chat_history=history_for_agent)
                answer, thoughts, validation, source = parse_agent_response(response_dict)
                
                history_for_agent.append(HumanMessage(content=query))
                history_for_agent.append(AIMessage(content=answer))
            
            finally:
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        logger.info(f"Successfully deleted temporary image file: {image_path}")
                    except PermissionError as e:
                        logger.warning(f"Could not remove {image_path}: {e}")
            
        elif query:
            history_doc_context = history_for_agent
            if current_medical_document:
                logger.info("Processing Follow-up Query for Document")
                history_doc_context = [HumanMessage(content=f"We are discussing this document:\n{current_medical_document}")] + history_for_agent
            else:
                logger.info("Processing Text RAG query for Medical domain")
            
            logger.info(f"Original Query: '{query}'")
            standalone_query = get_standalone_question(query, history_doc_context, llm)
            logger.info(f"Standalone Query: '{standalone_query}'")
            
            agent = rag_systems['medical']
            if not agent: 
                raise Exception("Medical RAG system is not loaded.")
            
            response_dict = agent.answer(standalone_query, chat_history=history_doc_context)
            answer, thoughts, validation, source = parse_agent_response(response_dict)

            history_for_agent.append(HumanMessage(content=query))
            history_for_agent.append(AIMessage(content=answer))

        else:
            raise ValueError("No query or file provided.")
            
    except Exception as e:
        logger.error(f"Error on /medical page: {e}", exc_info=True)
        answer = f"An error occurred: {e}"
        thoughts = traceback.format_exc()
    
    # ✅ DEHYDRATE history back to dicts
    session['medical_history'] = dehydrate_history(history_for_agent)
    
    # ✅ Save the response
    session['latest_medical_response'] = {
        'answer': answer, 
        'thoughts': thoughts, 
        'validation': validation, 
        'source': source
    }
    session.modified = True
    
    # ✅ ADD DEBUG LOG
    logger.info(f"💾 SAVED TO SESSION - Answer length: {len(answer)}, First 100 chars: {answer[:100]}")
    # --- MODIFIED FOR PYTHONANYWHERE ---
    # This should now show a real ID
    logger.info(f"💾 Session ID: {session.get('_id', 'NO ID')}") 
    # --- END MODIFICATION ---
    logger.info(f"💾 History length: {len(history_for_agent)}")
                             
    return redirect(url_for('medical_page'))

@app.route("/medical/clear")
def clear_medical_chat():
    session.pop('medical_history', None)
    session.pop('current_medical_document', None)
    logger.info("Medical chat history cleared.")
    return redirect(url_for('medical_page'))

@app.route("/islamic", methods=["GET", "POST"])
def islamic_page():
    #Use session
    
    if request.method == "GET":
        # Load all latest data from session (or default to empty if not found)
        latest_response = session.pop('latest_islamic_response', {}) # POP to clear it after one display
        
        answer = latest_response.get('answer', "")
        thoughts = latest_response.get('thoughts', "")
        validation = latest_response.get('validation', "")
        source = latest_response.get('source', "")
        
        # Clear history only when a user first navigates (no latest_response and no current history)
        if not latest_response and 'islamic_history' not in session:
            session.pop('islamic_history', None)

        # Hydrate the history from session dicts to LangChain objects
        raw_history_list = session.get('islamic_history', [])
        history = hydrate_history(raw_history_list)
        
        return render_template("islamic_page.html", 
                               history=history,
                               answer=answer,
                               thoughts=thoughts,
                               validation=validation,
                               source=source)
    
    # POST Request Logic
    answer, thoughts, validation, source = "", "", "", ""
    # Hydrate history first so it's a list of objects for the agent
    raw_history_list = session.get('islamic_history', [])
    history = hydrate_history(raw_history_list)
    
    # This try/except block wraps the ENTIRE POST logic
    try:
        query = standardize_query(request.form.get("query", ""))
        has_image = 'image' in request.files and request.files['image'].filename
        
        final_query = query # Default to the original query
        
        if has_image:
            logger.info("Processing Multimodal RAG query for Islamic domain")
            
            file = request.files['image']
            
            # --- MODIFIED FOR PYTHONANYWHERE ---
            global UPLOAD_DIR
            image_path = os.path.join(UPLOAD_DIR, file.filename)
            # --- END MODIFICATION ---
            
            try:
                file.save(image_path)
                file.close() 
                
                with open(image_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                
                vision_prompt = f"Analyze this image's main subject. User's query is: '{query}'"
                message = HumanMessage(content=[{"type": "text", "text": vision_prompt}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}])
                visual_prediction = llm.invoke([message]).content

                enhanced_query = (
                    f'User Query: "{query}" '
                    f'Context from an image provided by the LLM: "{visual_prediction}" '
                    'Based on the user\'s query and the context from LLM, provide a comprehensive answer.'
                )
                logger.info(f"Create enchanced query : {enhanced_query}")
                
                final_query = enhanced_query 
            
            finally:
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        logger.info(f"Successfully cleaned up {image_path}")
                    except PermissionError as e:
                        logger.warning(f"Could not remove {image_path} after processing. "
                                       f"File may be locked. Error: {e}")
            
        elif query: # Only run text logic if there's a query and no image
            logger.info("Processing Text RAG query for Islamic domain")
            standalone_query = get_standalone_question(query, history,llm)
            logger.info(f"Original Query: '{query}'")
            print(f"📚 Using chat history with {len(history)} previous messages to create standalone query")
            logger.info(f"Standalone Query: '{standalone_query}'")
            final_query = standalone_query
            
        if not final_query: 
            raise ValueError("No query or file provided.")
        
        agent = rag_systems['islamic']
        if not agent: raise Exception("Islamic RAG system is not loaded.")
        response_dict = agent.answer(final_query, chat_history=history)
        answer, thoughts , validation, source = parse_agent_response(response_dict)
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer))

    except Exception as e:
        logger.error(f"Error on /islamic page: {e}", exc_info=True)
        answer = f"An error occurred: {e}"
        thoughts = traceback.format_exc()
            
    # Save updated history and LATEST RESPONSE DATA back to the session
    session['islamic_history'] = dehydrate_history(history)
    session['latest_islamic_response'] = {
        'answer': answer, 
        'thoughts': thoughts, 
        'validation': validation, 
        'source': source
    }
    session.modified = True
    # --- ADD THIS DEBUG LINE ---
    logger.info(f"DEBUG: Saving to session: ANSWER='{answer[:50]}...', THOUGHTS='{thoughts[:50]}...'")                    
    logger.debug(f"Redirecting after saving latest response.")
    return redirect(url_for('islamic_page'))

@app.route("/islamic/clear")
def clear_islamic_chat():
    session.pop('islamic_history', None)
    logger.info("Islamic chat history cleared.")
    return redirect(url_for('islamic_page'))

@app.route("/insurance", methods=["GET", "POST"])
def insurance_page():
    if request.method == "GET" :
        latest_response = session.pop('latest_insurance_response',{})
        
        answer = latest_response.get('answer', "")
        thoughts = latest_response.get('thoughts', "")
        validation = latest_response.get('validation', "")
        source = latest_response.get('source', "")
        
        if not latest_response and 'insurance_history' not in session:
            session.pop('insurance_history', None)

        # Hydrate the history from session dicts to LangChain objects
        raw_history_list = session.get('insurance_history', [])
        history = hydrate_history(raw_history_list)
        
        return render_template("insurance_page.html", # You will need to create this HTML file
                                history=history,
                                answer=answer,
                                thoughts=thoughts,
                                validation=validation,
                                source=source)
    
    # POST Request Logic
    answer, thoughts, validation, source = "", "", "", ""
    # Hydrate history first so it's a list of objects for the agent
    raw_history_list = session.get('insurance_history', [])
    history = hydrate_history(raw_history_list)
    
    try:
        query = standardize_query(request.form.get("query", ""))
        
        if query:
            logger.info("Processing Text RAG query for Insurance domain")
            standalone_query = get_standalone_question(query, history, llm)
            logger.info(f"Original Query: '{query}'")
            logger.info(f"Standalone Query: '{standalone_query}'")
            
            agent = rag_systems['insurance']
            if not agent: raise Exception("Insurance RAG system is not loaded.")
            response_dict = agent.answer(standalone_query, chat_history=history)
            answer, thoughts, validation, source = parse_agent_response(response_dict)
            
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=answer))
        else:
            raise ValueError("No query provided.")

    except Exception as e:
        logger.error(f"Error on /insurance page: {e}", exc_info=True)
        answer = f"An error occurred: {e}"
        thoughts = traceback.format_exc()
            
    session['insurance_history'] = dehydrate_history(history)
    session['latest_insurance_response'] = {
        'answer': answer, 
        'thoughts': thoughts, 
        'validation': validation, 
        'source': source
    }
    session.modified = True
                        
    logger.debug(f"Redirecting after saving latest response.")
    return redirect(url_for('insurance_page'))

@app.route("/insurance/clear")
def clear_insurance_chat():
    session.pop('insurance_history', None)
    logger.info("Insurance chat history cleared.")
    return redirect(url_for('insurance_page'))

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route('/metrics/<domain>')
def get_metrics(domain):
    """API endpoint to get metrics for a specific domain."""
    try:
        if domain == "medical" and rag_systems['medical']:
            stats = rag_systems['medical'].metrics_tracker.get_stats()
        elif domain == "islamic" and rag_systems['islamic']:
            stats = rag_systems['islamic'].metrics_tracker.get_stats()
        elif domain == "insurance" and rag_systems['insurance']:
            stats = rag_systems['insurance'].metrics_tracker.get_stats()
        elif not rag_systems.get(domain):
            return jsonify({"error": f"{domain} RAG system not loaded"}), 500
        else:
            return jsonify({"error": "Invalid domain"}), 400
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics/reset/<domain>', methods=['POST'])
def reset_metrics(domain):
    """Reset metrics for a domain (useful for testing)."""
    try:
        if domain == "medical" and rag_systems['medical']:
            rag_systems['medical'].metrics_tracker.reset_metrics()
        elif domain == "islamic" and rag_systems['islamic']:
            rag_systems['islamic'].metrics_tracker.reset_metrics()
        elif domain == "insurance" and rag_systems['insurance']:
            rag_systems['insurance'].metrics_tracker.reset_metrics()
        elif not rag_systems.get(domain):
            return jsonify({"error": f"{domain} RAG system not loaded"}), 500
        else:
            return jsonify({"error": "Invalid domain"}), 400
        
        return jsonify({"success": True, "message": f"Metrics reset for {domain}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask app for deployment testing...")
    # This port 7860 is what Hugging Face Spaces expects by default
    app.run(host="0.0.0.0", port=7860, debug=False)