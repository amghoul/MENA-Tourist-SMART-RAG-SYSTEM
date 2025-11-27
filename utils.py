import os
from langgraph.graph import StateGraph
from typing import Any

def visualize_langgraph_workflow(graph_app: Any, filename: str = "rag_workflow_graph", output_dir: str = "."):
    """
    Generates a visual representation (PNG) of the LangGraph workflow.

    Note: This requires the 'graphviz' library to be installed:
    pip install graphviz pydot
    
    Additionally, the 'graphviz' system dependency must be installed 
    (e.g., 'sudo apt-get install graphviz' on Debian/Ubuntu, or using your system's package manager).

    Args:
        graph_app: The compiled LangGraph application instance (e.g., RAGAgent.app).
        filename: The base name for the output file (e.g., "rag_workflow_graph").
        output_dir: The directory where the file should be saved.
    """
    full_path = os.path.join(output_dir, filename + ".png")
    
    print("-" * 50)
    print(f"🎨 Attempting to visualize LangGraph workflow...")
    
    try:
        # LangGraph's compiled graph provides a get_graph().draw() utility
        graph_app.get_graph().draw(full_path, prog="dot", args='-Gdpi=300')
        print(f"✅ Workflow visualized successfully!")
        print(f"   Image saved to: {full_path}")
    except ImportError:
        print("❌ Visualization Failed: The 'graphviz' and/or 'pydot' Python packages are not installed.")
        print("   Please install them: pip install graphviz pydot")
    except Exception as e:
        print(f"❌ An error occurred during visualization: {e}")
        print("   Ensure the 'graphviz' system package is installed and accessible in your environment.")

    print("-" * 50)