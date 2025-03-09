from langgraph.graph import StateGraph, START, END
from state import ProcurementState
from nodes import proposal_processor  # Importing the tool

# ✅ Define graph with a meaningful name
rfp_analysis_workflow = StateGraph(ProcurementState)

rfp_analysis_workflow.add_edge(START, "ProposalProcessor")

# ✅ Add ProposalProcessor node
rfp_analysis_workflow.add_node("ProposalProcessor", proposal_processor)

# ✅ Define edges (Link START → ProposalProcessor → END)
rfp_analysis_workflow.add_edge("ProposalProcessor", END)

# ✅ Compile graph
graph = rfp_analysis_workflow.compile()
