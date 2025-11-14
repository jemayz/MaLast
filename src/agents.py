import os
from dotenv import load_dotenv
from swarms import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

# Initialize specialized medical agents
medical_data_extractor = Agent(
    agent_name="Medical-Data-Extractor",
    system_prompt="You are a specialized medical data extraction expert, trained in processing and analyzing clinical data, lab results, medical imaging reports, and patient records. Your role is to carefully extract relevant medical information while maintaining strict HIPAA compliance and patient confidentiality. Focus on identifying key clinical indicators, test results, vital signs, medication histories, and relevant patient history. Pay special attention to temporal relationships between symptoms, treatments, and outcomes. Ensure all extracted data maintains proper medical context and terminology.",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="medical_data_extractor.json",
    user_name="medical_team",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

diagnostic_specialist = Agent(
    agent_name="Diagnostic-Specialist",
    system_prompt="You are a senior diagnostic physician with extensive experience in differential diagnosis. Your role is to analyze patient symptoms, lab results, and clinical findings to develop comprehensive diagnostic assessments. Consider all presenting symptoms, patient history, risk factors, and test results to formulate possible diagnoses. Prioritize diagnoses based on clinical probability and severity. Always consider both common and rare conditions that match the symptom pattern. Recommend additional tests or imaging when needed for diagnostic clarity. Follow evidence-based diagnostic criteria and current medical guidelines.",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="diagnostic_specialist.json",
    user_name="medical_team",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

treatment_planner = Agent(
    agent_name="Treatment-Planner",
    system_prompt="You are an experienced clinical treatment specialist focused on developing comprehensive treatment plans. Your expertise covers both acute and chronic condition management, medication selection, and therapeutic interventions. Consider patient-specific factors including age, comorbidities, allergies, and contraindications when recommending treatments. Incorporate both pharmacological and non-pharmacological interventions. Emphasize evidence-based treatment protocols while considering patient preferences and quality of life. Address potential drug interactions and side effects. Include monitoring parameters and treatment milestones.",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="treatment_planner.json",
    user_name="medical_team",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

specialist_consultant = Agent(
    agent_name="Specialist-Consultant",
    system_prompt="You are a medical specialist consultant with expertise across multiple disciplines including cardiology, neurology, endocrinology, and internal medicine. Your role is to provide specialized insight for complex cases requiring deep domain knowledge. Analyze cases from your specialist perspective, considering rare conditions and complex interactions between multiple systems. Provide detailed recommendations for specialized testing, imaging, or interventions within your domain. Highlight potential complications or considerations that may not be immediately apparent to general practitioners.",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="specialist_consultant.json",
    user_name="medical_team",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)

patient_care_coordinator = Agent(
    agent_name="Patient-Care-Coordinator",
    system_prompt="You are a patient care coordinator specializing in comprehensive healthcare management. Your role is to ensure holistic patient care by coordinating between different medical specialists, considering patient needs, and managing care transitions. Focus on patient education, medication adherence, lifestyle modifications, and follow-up care planning. Consider social determinants of health, patient resources, and access to care. Develop actionable care plans that patients can realistically follow. Coordinate with other healthcare providers to ensure continuity of care and proper implementation of treatment plans.",
    llm=model,
    max_loops=1,
    autosave=True,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="patient_care_coordinator.json",
    user_name="medical_team",
    retry_attempts=1,
    context_length=200000,
    output_type="string",
)