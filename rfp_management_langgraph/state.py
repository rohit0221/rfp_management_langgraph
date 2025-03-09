from typing import TypedDict, Dict

class ProcurementState(TypedDict):
    input_files: Dict[str, str]  # Tracks {step_name: file_path}
    output_files: Dict[str, str]  # Tracks {step_name: output_file_path}
    steps: Dict[str, str]  # Tracks {step_name: status}, e.g., "completed" / "pending"
