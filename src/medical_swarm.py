import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import traceback
import os

#1. SETUP
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load env
load_dotenv()
google_api_key=os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variable.Make sure it is in .env file")

# Agent Definition
NO_SUGGESTIONS_INSTRUCTION= (
    "This is a finalized medical report.Do not provide suggestions or improvements."
    "Focus strictly on your assigned task based on the provided data"
)
class MedicalAgent:
    def __init__(self, llm, name:str, role_prompt: str):
        self.llm = llm
        self.name = name
        self.role_prompt= role_prompt
    
    def run(self, input_data:str):
        
        try:
            logger.info(f"Agent '{self.name}' is processing")
            full_prompt=f"{NO_SUGGESTIONS_INSTRUCTION}\n\n{self.role_prompt}\n\n{input_data}"
            response=self.llm.invoke(full_prompt)
            logger.info(f"Agent {self.name} finished")
            return response.content
        except Exception as e:
            logger.error(f"Agent {self.name} error: {str(e)}")
            traceback.print_exc()
            return f"Error in agent {self.name}: {str(e)}"

# Initialize SWARM
llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro",temperature=0.1,google_api_key=google_api_key)

#Define specific roles for each agent in the team
medical_data_extractor = MedicalAgent(
    llm, "Medical Data Extractor",
    "You are a specialized medical data extraction expert. Your role is to extract relevant medical information, focusing on key clinical indicators, test results, vital signs, and patient history from the provided text."
    )
diagnostic_specialist = MedicalAgent(
    llm, "Diagnostic Specialist",
    "You are a senior diagnostic physician. Your role is to analyze the provided symptoms, lab results, and clinical findings to develop a diagnostic assessment based solely on the data."
    )
treatment_planner = MedicalAgent(
    llm, "Treatment Planner",
    "You are an experienced clinical treatment specialist. Your role is to outline a prescribed treatment plan (pharmacological and non-pharmacological interventions) based on the provided diagnosis and data."
    )
specialist_consultant = MedicalAgent(
    llm, "Specialist Consultant",
    "You are a medical specialist consultant. Your role is to provide deep insights on the existing diagnosis and treatment plan, highlighting any potential complications or considerations based on the record."
    )
patient_care_coordinator = MedicalAgent(
    llm, "Patient Care Coordinator (Orchestrator)",
    "You are a patient care coordinator specializing in comprehensive healthcare management. Your primary role is to manage a team of specialist agents and synthesize their findings."
    )

# ORCHESTRATOR LOGIC
def run_medical_swarm(document_text: str, initial_query: str,chat_history: list = None):
    """
    Orchestrates a swarm of medical agents to analyze document.
    
    Args:
        document_text: The full text of medical record
        initial_query: The initial question or goal of analysis
    
    Returns:
        A string containing final, synthesized response
    """
    logger.info("--- MEDICAL SWARM INITIATED ---")
    
    # If there's a history, use it as the starting point. Otherwise, start fresh.
    workspace = [f"Initial Patient Document:\n{document_text}", f"Initial Goal: '{initial_query}'"]
    
    # The Patient Care Coordinator is our orchestrator/manager
    orchestrator= patient_care_coordinator
    
    # Limit the number of collab rounds to prevent infinite loops
    for i in range(5):
        logger.info(f"\n-- Swarm Iteration {i+1} --")
        current_state = "\n\n".join(workspace)
        
        # The orchestrator reviews the current state and decide the next action.
        orchestrator_prompt= f"""
        You are the Patient Care Coordinator managing a team of specialist AI agents to analyze a medical case.
        Review the current Case File below and decide the single next best action.

        Your available specialists are:
        - 'medical_data_extractor': Best for the first step to get raw data from the initial document.
        - 'diagnostic_specialist': Best for forming a diagnosis after key data has been extracted.
        - 'treatment_planner': Best for creating a plan after a clear diagnosis is available.
        - 'specialist_consultant': Best for getting deeper insight on an existing diagnosis or plan.

        Case File (Summary of work so far):
        ---
        {current_state}
        ---

        Based on the file, which specialist should be called next? Or, if you have a clear diagnosis, treatment plan, and specialist insight, is it time to write the final summary for the patient?

        Respond with ONLY one of the following commands:
        - "CALL: medical_data_extractor"
        - "CALL: diagnostic_specialist"
        - "CALL: treatment_planner"
        - "CALL: specialist_consultant"
        - "FINISH"
        """
        command=orchestrator.run(orchestrator_prompt).strip().upper()       
        logger.info(f"Orchestrator Command: {command}")
        
        if command == "CALL: MEDICAL_DATA_EXTRACTOR":
            report= medical_data_extractor.run(f"Original Document:\n{document_text}")
            workspace.append(f"--- Extractor's Report ---\n{report}")
            
        elif command == "CALL: DIAGNOSTIC_SPECIALIST":
            # Other agents get the whole workspace for context
            report = diagnostic_specialist.run(f"Full Case File for Diagnosis:\n{current_state}")
            workspace.append(f"--- Diagnostician's Report ---\n{report}")
            
        elif command == "CALL: TREATMENT_PLANNER":
            report = treatment_planner.run(f"Full Case File for Treatment Plan:\n{current_state}")
            workspace.append(f"--- Treatment Planner's Report ---\n{report}")
            
        elif command == "CALL: SPECIALIST_CONSULTANT":
            report = specialist_consultant.run(f"Full Case File for Specialist Consultation:\n{current_state}")
            workspace.append(f"--- Specialist Consultant's Report ---\n{report}")
            
        elif command == "FINISH":
            logger.info("Orchestrator has decided the work is complete. Generating final summary.")
            final_summary_prompt = f"You are the Patient Care Coordinator. Based on the complete case file below, write a comprehensive, patient-facing summary that coordinates all the findings.\n\nFull Case File:\n{current_state}"
            final_answer = orchestrator.run(final_summary_prompt)
            return final_answer
        else:
            logger.warning(f"Orchestrator gave an unknown command: '{command}'. Ending swarm.")
            break
        
    # Fallback if the loop finishes without a "FINISH" command
    logger.warning("Swarm reached max iterations. Finalizing with current data.")
    final_fallback_prompt = f"You are the Patient Care Coordinator. The analysis time has expired. Summarize the findings from the case file below into a cohesive patient-facing report.\n\nFull Case File:\n{current_state}"
    final_answer = orchestrator.run(final_fallback_prompt)
    return final_answer        
