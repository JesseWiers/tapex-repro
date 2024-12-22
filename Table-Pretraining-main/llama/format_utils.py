import re

def transform_table_to_markdown(table_str):
    # Remove any leading or trailing whitespace
    table_str = table_str.strip()
    
    # Find 'col :' and extract everything after it
    col_match = re.search(r'col\s*:', table_str)
    if not col_match:
        raise ValueError("No column headers found in input.")
    col_start = col_match.end()
    rest_str = table_str[col_start:].strip()
    
    # Split the rest of the string into header and rows using 'row x :' as delimiter
    # The pattern looks for 'row' followed by any number and a colon
    row_pattern = r'row\s*\d+\s*:'
    splits = re.split(row_pattern, rest_str)
    
    if len(splits) < 2:
        raise ValueError("No data rows found in input.")
    
    # The first split is the header
    header_str = splits[0].strip()
    headers = [h.strip() for h in header_str.split('|')]
    
    # Create the header separator
    header_separator = ['-' * max(len(h), 3) for h in headers]
    
    # The rest are the data rows
    # Use regex to find all occurrences of 'row x :' to get row labels (for ordering if needed)
    row_matches = re.findall(row_pattern, rest_str)
    
    # Build the list of data rows
    table_rows = []
    for row_str in splits[1:]:
        row_data = row_str.strip()
        if row_data:
            cells = [c.strip() for c in row_data.split('|')]
            table_rows.append(cells)
    
    # Build the markdown table
    markdown_lines = []
    # Header row
    markdown_lines.append('| ' + ' | '.join(headers) + ' |')
    # Header separator
    markdown_lines.append('| ' + ' | '.join(header_separator) + ' |')
    # Data rows
    for row in table_rows:
        markdown_lines.append('| ' + ' | '.join(row) + ' |')
    
    markdown_table = '\n'.join(markdown_lines)
    return markdown_table
