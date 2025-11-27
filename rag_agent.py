import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
import re
import time
from typing_extensions import TypedDict
# Note: OllamaLLM is used as a placeholder. Replace with your preferred LLM provider.
from langchain_ollama import OllamaLLM 


# --- AGENT STATE DEFINITION ---
class AgentState(TypedDict):
    """
    Represents the state of our RAG agent's execution flow.
    
    This TypedDict is used by LangGraph to manage the data flow between nodes.

    Attributes:
        question (str): The initial or current user question (can be rewritten).
        Answer (str): The final generated answer.
        docs (list[str]): The list of retrieved text chunks (context).
        historical_questions (list[str]): A list of past questions (used by the rewriter).
    """
    question: str
    Answer: str
    docs: list[str]
    historical_questions: list[str]


class RAGAgent:
    """
    A RAG Agent that uses a Retrieval-Augmented Generation pipeline 
    with a conditional rewrite step for better query performance on Arab Heritage data.
    """
    def __init__(self):
        # Initialize the Sentence Transformer model for embedding
        # Using a multilingual model suitable for both English and Arabic
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu')
        
        # Load the FAISS index and the corresponding text chunks/metadata
        try:
            self.index = faiss.read_index("heritage.index")
        except Exception:
            print("❌ Error: 'heritage.index' not found. Ensure you run collect_data.py first.")
            exit()

        # Initialize the LLM (using the specified Ollama model)
        self.llm = OllamaLLM(model="mistral:7b", temperature=0.0) 
        
        self.chunks = []
        self.metadata = []

        self.read_file("heritage.pkl")
        
        # Compile the LangGraph application
        self.app = self.build_graph()


    def read_file(self, file="heritage.pkl"):
        """Loads chunks and metadata from the pickle file."""
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            print(f"✅ Loaded {len(self.chunks)} chunks")
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
            print("Ensure 'heritage.pkl' exists after running collect_data.py.")
            exit()


    def retriever_node(self, state: AgentState) -> dict:
        """
        Retrieves relevant documents from the FAISS index based on the question.
        
        This is the "R" (Retrieval) step of the RAG process.
        """
        question = state["question"]
        print(f"--- 🔍 RETRIEVING for question: {question[:50]}... ---")

        # 1. Embed the question
        question_embedding = self.model.encode(question)

        # 2. Search the FAISS index (k=4 for the top 4 chunks)
        # D: distances, I: indices
        D, I = self.index.search(question_embedding.reshape(1, -1), k=4)
        
        # 3. Get the corresponding text chunks
        retrieved_chunks = [self.chunks[i] for i in I[0]]
        
        # 4. Filter and prepare chunks for the LLM
        unique_chunks = list(dict.fromkeys(retrieved_chunks))

        # Update the state with the retrieved documents
        print(f"--- ✅ RETRIEVED {len(unique_chunks)} chunks ---")
        return {"docs": unique_chunks}


    def grader_node(self, state: AgentState) -> dict:
        """
        Grades the retrieved documents against the question to see if they are sufficient.
        
        This decision step determines if we can proceed to generate an answer
        or if the question needs to be rewritten for better retrieval.
        """
        question = state["question"]
        docs = state["docs"]
        
        print(f"--- 🧐 GRADING CONTEXT for question: {question[:50]}... ---")

        # Grade prompt
        prompt = PromptTemplate(
            template="""
            You are a context grader. Your task is to analyze the following question and the retrieved documents.
            Determine if the documents contain enough information to fully and accurately answer the question.

            If the documents are highly relevant and sufficient, respond with 'generate'.
            If the documents are only partially relevant or insufficient, respond with 'rewrite'.
            If the question is completely unanswerable based on the context (e.g., about a non-heritage topic), respond with 'end'.

            Question: {question}
            --- Retrieved Documents ---
            {docs}
            --------------------------
            Decision (generate, rewrite, or end):
            """,
            input_variables=["question", "docs"],
        )

        chain = prompt | self.llm | StrOutputParser()
        
        # Invoke the grader LLM
        raw_decision = chain.invoke({"question": question, "docs": "\n---\n".join(docs)})
        decision_str = raw_decision.strip().lower()
        
        # Clean up the decision string to ensure it's one of the valid states
        if "generate" in decision_str:
            final_decision = "generate"
        elif "rewrite" in decision_str:
            final_decision = "rewrite"
        elif "end" in decision_str:
            final_decision = "end"
        else:
            # Fallback for unexpected LLM output
            final_decision = "generate" 
            print(f"--- ⚠️ GRADER FAILED, defaulting to: {final_decision} (Raw: {raw_decision[:20]}...) ---")

        print(f"--- ➡️ GRADER DECISION: {final_decision} ---")
        # Ensure we return a dictionary with the expected key for state update
        return {"decision": final_decision}


    def decision_node(self, state: AgentState):
        """Routes the execution based on the grader's decision."""
        # Use .get() with a default value to prevent the KeyError if the decision key is missing
        decision = state.get("decision", "generate")
        return decision


    def rewriter_node(self, state: AgentState) -> dict:
        """
        Rewrites the user's question to improve retrieval accuracy.
        
        This step helps handle complex, ambiguous, or conversational queries.
        """
        question = state["question"]
        historical_questions = state["historical_questions"]
        docs = state["docs"]
        
        print(f"--- 📝 REWRITING QUESTION: {question[:50]}... ---")

        # Rewriter prompt
        prompt = PromptTemplate(
            template="""
            You are a question rewriter. Your goal is to rewrite the user's question to be more specific, 
            or to incorporate missing context from the previous turn, or to be better optimized for 
            keyword search on the provided documents.

            If the question is conversational (e.g., "what about it?"), use the Historical Questions 
            to create a standalone, specific query.

            Historical Questions: {historical_questions}
            Original Question: {question}
            Context Retrieved So Far: {docs}

            Rewrite the question to be more effective for search retrieval. 
            New Question: 
            """,
            input_variables=["question", "historical_questions", "docs"],
        )

        chain = prompt | self.llm | StrOutputParser()
        
        # Invoke the rewriter LLM
        new_question = chain.invoke({
            "question": question, 
            "historical_questions": "\n".join(historical_questions),
            "docs": "\n---\n".join(docs)
        }).strip()

        # Update historical questions for the next cycle
        historical_questions.append(question)
        
        print(f"--- 🔄 REWRITTEN QUESTION: {new_question[:50]}... ---")
        return {"question": new_question, "historical_questions": historical_questions}


    def generate_node(self, state: AgentState) -> dict:
        """
        Generates the final answer using the question and the retrieved context.
        
        This is the "G" (Generation) step of the RAG process.
        """
        question = state["question"]
        docs = state["docs"]
        
        print(f"--- 🧠 GENERATING ANSWER for question: {question[:50]}... ---")

        # Generator prompt
        prompt = PromptTemplate(
            template="""
            You are a helpful and knowledgeable expert on Arab cultural heritage sites.
            Your task is to answer the user's question based ONLY on the provided context.
            
            1. Use all relevant information from the context.
            2. If the answer is not found in the context, state clearly, "I apologize, but I couldn't find the answer to that question in the available heritage documents."
            3. Answer in the same language as the user's question.

            Question: {question}
            --- Context ---
            {docs}
            ---------------
            Answer:
            """,
            input_variables=["question", "docs"],
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Invoke the generator LLM
        answer = chain.invoke({"question": question, "docs": "\n---\n".join(docs)}).strip()
        
        print(f"--- ✅ GENERATION COMPLETE ---")
        return {"Answer": answer}


    def build_graph(self):
        """Builds and compiles the RAG LangGraph state machine."""
        # The graph structure: Retriever -> Grader (Conditional) -> [Generate | Rewriter -> Retriever]
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("retriever", self.retriever_node)
        graph.add_node("grader", self.grader_node)
        graph.add_node("generate", self.generate_node)
        graph.add_node("rewriter", self.rewriter_node)

        # Set the entry point
        graph.set_entry_point("retriever")

        # Define edges
        graph.add_edge("retriever", "grader")

        # Conditional routing from grader
        graph.add_conditional_edges(
            "grader",
            self.decision_node,
            {
                "generate": "generate", # Go to generate if context is sufficient
                "rewrite": "rewriter",  # Go to rewriter if context is insufficient
                "end": END              # End if question is unanswerable
            }
        )

        # Loop back from rewriter to retriever with the new question
        graph.add_edge("rewriter", "retriever")

        # End the process after generation
        graph.add_edge("generate", END)

        return graph.compile()


    def ask(self, question: str):
        """Ask a question using the RAG agent (non-streaming, simple output)."""
        result = self.app.invoke({
            "question": question,
            "Answer": "",
            "docs": [],
            "historical_questions": []
        })
        return result['Answer']


    def ask_and_stream_steps(self, question: str):
        """
        Asks a question and yields the output of each node execution for step-by-step visibility.
        """
        inputs = {
            "question": question,
            "Answer": "",
            "docs": [],
            "historical_questions": []
        }
        
        # Use stream() to get output from each node
        for step in self.app.stream(inputs):
            # Check for END state transition
            if END in step:
                node_name = "END"
                output = step[END]
            else:
                node_name = next(iter(step))
                output = step[node_name]
            
            # Skip printing steps that have no output (like the END transition)
            if output is None:
                continue

            # Print state changes for visibility
            print("-" * 50)
            print(f"*** STEP: {node_name.upper()} ***")
            
            # Safely check for keys in the output dictionary
            if isinstance(output, dict):
                if 'question' in output:
                    print(f"Question: {output['question'][:80]}...")
                if 'docs' in output:
                    print(f"Retrieved {len(output['docs'])} chunks.")
                if 'decision' in output:
                    print(f"Decision: {output['decision']}")
                if 'Answer' in output:
                    # We show the full answer when it's the final output
                    pass 
            print("-" * 50)
            
            yield node_name, output

        # The final answer is retrieved using invoke after the stream completes
        # Note: The stream itself doesn't return the final state, so we invoke again 
        # or use the full state history to get the last state. 
        # Since you want the final answer, let's keep the final invoke.
        final_state = self.app.invoke(inputs)
        return final_state['Answer']
        """
        Asks a question and yields the output of each node execution for step-by-step visibility.
        """
        inputs = {
            "question": question,
            "Answer": "",
            "docs": [],
            "historical_questions": []
        }
        
        # Use stream() to get output from each node
        for step in self.app.stream(inputs):
            node_name = next(iter(step))
            output = step[node_name]
            
            # Print state changes for visibility
            print("-" * 50)
            print(f"*** STEP: {node_name.upper()} ***")
            if 'question' in output:
                print(f"Question: {output['question'][:80]}...")
            if 'docs' in output:
                print(f"Retrieved {len(output['docs'])} chunks.")
            if 'decision' in output:
                print(f"Decision: {output['decision']}")
            if 'Answer' in output:
                # We show the full answer when it's the final output
                pass 
            print("-" * 50)
            
            yield node_name, output

        return self.app.invoke(inputs)['Answer']