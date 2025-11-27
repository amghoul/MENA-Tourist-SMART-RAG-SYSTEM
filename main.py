import time

# Import the main agent class
from rag_agent import RAGAgent

# Import the testing function from the dedicated test file
from test_agent import check_accuracy_on_test_set 

# --- NEW IMPORT ---
from utils import visualize_langgraph_workflow
# ------------------


if __name__ == "__main__":
    # Initialize the RAG Agent
    agent = RAGAgent()
    
    # Update prompt to include the new command
    print("\n🤖 RAG Agent Ready! Ask questions about Arab heritage, or run the accuracy test by typing 'test' or visualize the workflow with 'graph'.\n")

    while True:
        # Update input prompt
        question = input("❓ Your question (or 'test' / 'graph' / 'quit'): ")
        
        if question.lower() == 'quit':
            break
        elif question.lower() == 'test':
            # Run the imported test suite
            check_accuracy_on_test_set(agent)
        # --- NEW COMMAND LOGIC ---
        elif question.lower() == 'graph':
            # Visualize the LangGraph workflow. The graph is the agent's internal app.
            visualize_langgraph_workflow(agent.app)
        # -------------------------
        elif question.strip():
            # Run the streaming function to show the RAG execution path
            print("\n--- RAG EXECUTION STEPS ---")
            final_answer = ""
            # Note: We capture the final answer by streaming until the END state
            for node, output in agent.ask_and_stream_steps(question):
                if node == 'generate' and 'Answer' in output:
                    final_answer = output['Answer']

            if final_answer:
                 print("\n" + "=" * 70)
                 print("FINAL ANSWER")
                 print("=" * 70)
                 print(final_answer)
                 print("=" * 70 + "\n")
        else:
            print("Please enter a question, 'test', 'graph', or 'quit'.")