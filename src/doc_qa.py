from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.retrievers import BM25Retriever
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_core.agents import AgentAction
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from flashrank import Ranker, RerankRequest
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
# from src.metrics_tracker import MetricsTracker
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class QA:
    def __init__(self, retriever):
        self.system_template = """
        Answer the user's questions based on the below context. 
        If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":
        <context>
        {context}
        </context>
        """
        self.question_answering_prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_template),
             MessagesPlaceholder(variable_name="messages")]
        )
        self.retriever = retriever
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.qa_chain = create_stuff_documents_chain(self.llm, self.question_answering_prompt)

    def query(self):
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            docs = self.retriever.invoke(query)
            response = self.qa_chain.invoke(
                {"context": docs,
                 "messages": [HumanMessage(content=query)]}
            )
            print(f"AI: {response}")

class RetrievalAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    def deduplicate_context(self, context_list): 
        """Deduplicate context entries to avoid repetition."""
        seen = set()
        deduped = []
        for item in context_list:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return "\n".join(deduped) if deduped else "No relevant context found."

    def retrieve(self, query, top_k=5):
        """
        Retrieve the top-k relevant contexts from ChromaDB based on the query.
        
        Args:
            query (str): The query or prediction to search for.
            top_k (int): Number of top results to return (default: 3).
        
        Returns:
            str: Deduplicated context string from the top-k results.
        """
        logger.info(f"Retrieving for query: {query}")
        try:
            # Perform similarity search using ChromaDB retriever
            results = self.retriever.invoke(query, k=top_k) 
            logger.info(f"Retrieved documents: {[doc.metadata.get('source') for doc in results]}")
            
            # Extract the page content (context) from each result
            contexts = [doc.page_content for doc in results]
            
            # Deduplicate the contexts
            deduped_context = self.deduplicate_context(contexts)
            logger.info(f"Deduplicated context: {deduped_context}")
            
            return deduped_context
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return "Retrieval failed due to error."

class AgenticQA:
    def __init__(self, config=None):  
        logger.info("Initializing AgenticQA...")
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "IMPORTANT: DO NOT provide any answers or explanations. ONLY rephrase the question if needed. "
            "If the question is already clear and standalone, return it exactly as is. "
            "Output ONLY the reformulated question, nothing else."
        )
        
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", self.contextualize_q_system_prompt),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}")]
        )
        self.qa_system_prompt = (
            "You are an assistant that answers questions in a specific domain for citizens mainly in Malaysia, "
            "depending on the context. "
            "You will receive:\n"
            "  • domain = {domain}  (either 'medical', 'islamic' , or 'insurance')\n"
            "  • context = relevant retrieved passages\n"
            "  • user question\n\n"
            "Instructions based on domain:\n"
            "1. If domain = 'medical' :\n"
            "   - Answer the question in clear, simple layperson language, "
            "   - Citing your sources (e.g. article name, section)."
            "   - Add a medical disclaimer: “I am not a doctor…”.\n"
            "2. If domain = 'islamic':\n"
            "   - **ALWAYS present both Shafi'i AND Maliki perspectives** if the question is about fiqh/rulings\n"
            "   - **Cite specific sources**: Always mention the book name (e.g., 'According to Muwatta Imam Malik...', 'Minhaj al-Talibin states...', 'Umdat al-Salik explains...')\n"
            "   - **Structure answer as**:\n" 
            "      - Shafi'i view (from Umdat al-Salik/Minhaj): [ruling with citation]\n"
            "      - Maliki view (from Muwatta): [ruling with citation]\n"
            "      - If they agree: mention the consensus\n"
            "      - If they differ: present both views objectively without favoring one\n"
            "   - **For hadith questions**: provide the narration text, source (book name, hadith number), "
            "       and authenticity grading if available\n"
            "   - If the context does not contain relevant information from BOTH madhabs, acknowledge which sources you have "
            "      (e.g., 'Based on Shafi'i sources only...') and suggest consulting additional madhab resources.\n"
            "   - **Always end with**: 'This is not a fatwa. Consult a local scholar for guidance specific to your situation.'\n"
            "   - Keep answers concise but comprehensive enough to show different scholarly views.\n\n"
            
            "3. If domain = 'insurance':\n"
            "   - Your knowledge is STRICTLY limited to Etiqa Takaful (Motor and Car policies).\n"
            "   - First, try to answer ONLY using the provided <context>.\n"
            "   - **If the answer is not in the context, YOU MUST SAY 'I do not have information on that specific topic.'** Do not make up an answer.\n"
            "   - If the user asks about other Etiqa products (e.g., medical, travel), you MUST use the 'EtiqaWebSearch' tool.\n"
            "   - If the user asks about another insurance company (e.g., 'Prudential', 'Takaful Ikhlas'), state that you can only answer about Etiqa Takaful.\n"
            "   - If the user asks a general insurance question (e.g., 'What is takaful?', 'What is an excess?'), use the 'GeneralWebSearch' tool.\n"
                    
            "4. For ALL domains: If the context does not contain the answer, do not make one up. Be honest.\n\n"
            "Context:\n"
            "{context}"
            )

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [("system", self.qa_system_prompt),
             MessagesPlaceholder("chat_history"),
             ("human", "{input}")]
        )
        self.react_docstore_prompt = hub.pull("aallali/react_tool_priority")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
        self.answer_validator = AnswerValidatorAgent(self.llm)

        self.retriever = None
        self.agent_executor = None
        self.tools = [] # Initialize the attribute
        self.domain = "general"
        self.answer_validator = None

        if config:
            logger.info(f"Configuring AgenticQA with provided config: {config}")
            try:
                collection_name = config["retriever"]["collection_name"]
                persist_directory = config["retriever"]["persist_directory"]
                self.domain = config.get("domain", "general") # Get domain from config
                
                # 1. Initialize the embedding function
                embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

                # 2. Connect to the persistent ChromaDB
                db_client = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding_function,
                    collection_name=collection_name
                )

                # 3. Set the retriever for this instance
                self.retriever = db_client.as_retriever()
                logger.info(f"✅ Successfully created retriever for collection '{collection_name}'")
                # Initialize validator *after* setting domain
                self.answer_validator = AnswerValidatorAgent(self.llm, self.domain)
                self._initialize_agent()
                
            except Exception as e:
                logger.error(f"❌ Error during AgenticQA setup for '{self.domain}': {e}", exc_info=True)
        else:
            logger.warning("⚠️ AgenticQA initialized without a config. Retriever will be None.")

    def _initialize_agent(self):
        """Build the ReAct agent"""
        """A helper function to build the agent components."""

        logger.info(f"Initializing agent for domain: '{self.domain}'")
        self.retrieval_agent = RetrievalAgent(self.retriever)
        
        # We need a RAG chain for the tool
        history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, self.contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        self.tools = [
            Tool(
                name="RAG",
                func=lambda query, **kwargs: rag_chain.invoke({
                    "input": query,
                    "chat_history": kwargs.get("chat_history", []),
                    "domain": self.domain
                })["answer"],
                description=(f"Use this tool first to search and answer questions about the {self.domain} domain using internal vector database.")
            )
            
        ]
        
        # --- DOMAIN-SPECIFIC TOOLS ---
        if self.domain == "insurance":
            # Add a specific tool for searching Etiqa's website
            etiqa_search_tool = TavilySearchResults(max_results=3)
            etiqa_search_tool.description = "Use this tool to search the Etiqa Takaful website for products NOT in the RAG context (e.g., medical, travel)."
            # This is a bit of a "hack" to force Tavily to search a specific site.
            # We modify the function it calls.
            original_etiqa_func = etiqa_search_tool.invoke
            def etiqa_site_search(query):
                return original_etiqa_func(f"site:etiqa.com.my {query}")
            
            self.tools.append(Tool(
                name="EtiqaWebSearch",
                func=etiqa_site_search,
                description=etiqa_search_tool.description
            ))
            
            # Add a general web search tool
            self.tools.append(Tool(
                name="GeneralWebSearch",
                func=TavilySearchResults(max_results=2).invoke,
                description="Use this tool as a fallback for general, non-Etiqa questions (e.g., 'What is takaful?')."
            ))
        else:
            # Medical and Islamic domains only get the general web search fallback
            self.tools.append(Tool(
                name="GeneralWebSearch",
                func=TavilySearchResults(max_results=2).invoke,
                description="Use this tool as a fallback if the RAG tool finds no relevant information or if the query is about a general topic."
            ))
        # --- END OF TOOL DEFINITION ---
        
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.react_docstore_prompt)
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5
        )
        logger.info(f"✅ Agent Executor(ReAct Agent) created successfully for '{self.domain}'.")
    
    def create_rag_chain(self, retriever):
        from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, self.contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def create_rag_agent(self):
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.react_docstore_prompt
        )

    def run(self, retriever,domain="general"):
        """This method is now used for manual/pipeline setup. app.py uses the config in __init__."""
        self.retriever = retriever
        self.domain = domain
        self._initialize_agent()
        
    def answer(self, query, chat_history=list):
        """
        Process a query using the agent and returns a clean dictionary
        Assumes the query is already standalone and receives chat history externally.
        """
        if not self.agent_executor:
            return {"answer": "Error: Agent not initialized.", "context": "", "validation": (False, "Init failed"), "source": "Error"}
        print(f"\n📝 AGENTIC_QA PROCESSING QUERY: '{query}'")
        
        response = self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history,
            "domain": self.domain # Pass domain to agent
        })
        thoughts= ""
        
        final_answer = response.get("output", "Could not generate an answer")
        
        tool_used = None
        if "intermediate_steps" in response:
            thought_log= []
            for step in response["intermediate_steps"]:
                # --- FIX: Unpack the (Action, Observation) tuple first ---
                action, observation = step
                
                # The logic inside the if block can now use 'action'
                if isinstance(action, AgentAction) and action.tool:
                    tool_used = action.tool #Capture the last tooused
                
                # Append Thought, Action, and Action Input (contained in action.log)
                thought_log.append(action.log)
                
                # CRITICAL FIX: The observation variable is now correctly defined
                thought_log.append(f"\nObservation: {str(observation)}\n---") 
            
            thoughts = "\n".join(thought_log)     

        # Assign source based on the LAST tool used
        if tool_used == "RAG":
            source = "Etiqa Takaful Database" if self.domain == "insurance" else "Domain Database (RAG)"
        elif tool_used == "EtiqaWebSearch":
            source = "Etiqa Website Search"
        elif tool_used == "GeneralWebSearch":
            source = "General Web Search"
        else:
            source = "Agent Logic"

        logger.info(f"Tool used: {tool_used}, Source determined: {source}")
        
        # Retrieve context only if the RAG tool was used
        context = self.retrieval_agent.retrieve(query) if source.endswith("(RAG)") or source.startswith("Etiqa Takaful Database") else "Web search results used."
        
        validation = self.answer_validator.validate(query, final_answer, source=source)
        
        return {"answer": final_answer, "context": context, "validation": validation, "source": source, "thoughts": thoughts}

class AnswerValidatorAgent:
    def __init__(self, llm,domain="general"):
        self.llm = llm
        self.domain = domain
        # General validation prompt for non-medical or WebSearch queries
        self.general_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an answer validator. Check if the generated answer is factually correct "
                "and relevant to the query. Return 'Valid' if the answer is correct and relevant, "
                "or 'Invalid: [reason]' if not, where [reason] is a brief explanation of the issue."
            )),
            ("human", "Query: {query}\nAnswer: {answer}")
        ])
        # Medical-specific validation prompt for RAG queries
        self.medical_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an answer validator. Check if the generated answer is factually correct, "
                "relevant to the query, and consistent with known medical knowledge. "
                "Return 'Valid' if the answer is correct and relevant, or 'Invalid: [reason]' if not, "
                "where [reason] is a brief explanation of the issue."
            )),
            ("human", "Query: {query}\nAnswer: {answer}")
        ])

    def validate(self, query, answer, source="RAG"):
        # --- NEW: Skip validation for insurance domain ---
        if self.domain == "insurance":
            logger.info(f"Skipping validation for insurance domain.")
            return True, "Validation skipped for insurance domain."
        # --- END OF NEW LOGIC ---
        try:
            # Use medical prompt for RAG responses, general prompt for WebSearch
            prompt = self.medical_prompt if source == "RAG" else self.general_prompt
            response = self.llm.invoke(prompt.format(query=query, answer=answer))
            validation = response.content.strip()
            logger.info(f"AnswerValidator result for query '{query}': {validation}")
            
            # Parse the response
            if validation.lower().startswith("valid"):
                return True, "Answer is valid and relevant."
            elif validation.lower().startswith("invalid"):
                reason = validation.split(":", 1)[1].strip() if ":" in validation else "No reason provided."
                return False, reason
            else:
                return False, "Validation response format unexpected."
        except Exception as e:
            logger.error(f"AnswerValidator error: {str(e)}")
            return False, "Validation failed due to error."