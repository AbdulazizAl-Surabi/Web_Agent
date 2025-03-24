import re
from tabulate import tabulate

def extract_final_action_content(text):
    """
    Extracts the extracted_content from the final ActionResult with is_done=True and success=True.
    It supports either single or double quotes.
    """
    pattern = (
        r"ActionResult\(is_done=True,\s*success=True,\s*extracted_content=(?P<quote>['\"])(?P<content>.*?)(?P=quote),\s*error=None,"
    )
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group("content")
    return None

def parse_final_content(content):
    """
    Parses the final content string.
    Expected format (each entry separated by one or more blank lines):
    
    1. URL: https://example.com/...
       Title: Some Article Title
       Summary: A brief summary of the content.

    Returns a list of rows: [ [number, url, title, summary], ... ]
    """
    # Split entries by two or more newlines
    blocks = re.split(r'\n\s*\n', content.strip())
    entries = []
    for block in blocks:
        # Try to extract number, URL, title, and summary from each block.
        m = re.search(
            r"(\d+)\.\s*URL:\s*(.*?)\n\s*Title:\s*(.*?)\n\s*Summary:\s*(.*)",
            block, re.DOTALL)
        if m:
            number = m.group(1)
            url = m.group(2).strip()
            title = m.group(3).strip()
            # Replace multiple whitespace and newlines inside the summary for readability
            summary = re.sub(r'\s+', ' ', m.group(4).strip())
            entries.append([number, url, title, summary])
    return entries

def main():
    # Read the output file (ensure output.txt is in the same directory)
    try:
        with open("output.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("File output.txt not found.")
        return
    
    final_content = extract_final_action_content(text)
    if not final_content:
        print("No final content found.")
        return
    
    entries = parse_final_content(final_content)
    if not entries:
        print("No entries could be parsed.")
        return

    # Display results in a table
    table = tabulate(entries, headers=["No", "URL", "Title", "Summary"], tablefmt="grid")
    print(table)

if __name__ == "__main__":
    main()
