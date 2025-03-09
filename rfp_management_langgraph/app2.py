import os
from graph import workflow_graph
from state import ProcurementState

def main():
    print("\n🚀 Running Procurement Workflow...\n")

    # ✅ Initialize state
    state: ProcurementState = {
        "input_files": {"proposal_pdfs": "./data/proposals/"},
        "output_files": {},
        "steps": {}
    }

    # ✅ Check if SKIP_VECTORIZER is set
    if os.getenv("SKIP_VECTORIZER") == "true":
        print("⚠️ Skipping ProposalProcessor (pre-marked as completed)")
        state["steps"]["ProposalProcessor"] = "completed"  # ✅ Trick the workflow

    # Invoke the graph
    result = workflow_graph.invoke(state)

    # Display the final workflow state
    print("\n✅ Final Graph Execution State:")
    print(result)

if __name__ == "__main__":
    main()
