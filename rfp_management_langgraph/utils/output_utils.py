import os

def save_markdown(content, filename):
    """
    Saves the retrieved supplier proposals to a Markdown file inside ./outputs.
    Ensures that content is converted to a string.
    """
    if not isinstance(content, str):  # ✅ Convert CrewOutput to string if needed
        content = str(content)

    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)  # ✅ Ensure ./outputs directory exists

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as md_file:
        md_file.write(content)

    print(f"\n Output saved to {output_path}")
