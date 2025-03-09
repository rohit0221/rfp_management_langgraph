from graph import graph
from state import ProcurementState

def main():
    print("\n🚀 Running Procurement Workflow...\n")

    # Initialize state with input directory
    state: ProcurementState = {
        "input_files": {"proposal_pdfs": "./data/proposals/"},  # Ensure this directory has PDFs
        "output_files": {},
        "steps": {}
    }

    # Invoke the graph
    result = graph.invoke(state)

    # Display the final workflow state
    print("\n✅ Final Graph Execution State:")
    print(result)

if __name__ == "__main__":
    main()
