import csv
import os
import re

def text_files_to_csv_with_headers(text_files, output_csv):
    """
    Combines multiple text files into one CSV file, with each file's lines as separate columns in each row.
    The filenames are used as column headers.

    Args:
        text_files (list): List of paths to text files.
        output_csv (str): Path to the output CSV file.
    """
    # Open the text files in read mode
    files = [open(file, "r", encoding="utf-8") for file in text_files]
    
    # Extract filenames for the header
    headers = [os.path.basename(file).split(".")[0] for file in text_files]

    # Open the output CSV file in write mode
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers to the CSV
        writer.writerow(headers)
        
        # Iterate over lines from each file simultaneously
        while True:
            # Read one line from each file
            lines = [f.readline().strip() for f in files]
            
            # Check if all files have reached EOF
            if all(line == "" for line in lines):
                break
            
            # Write the lines as a row in the CSV
            writer.writerow(lines)

    # Close all input files
    for f in files:
        f.close()



# Example usage
text_files = ["cleaned_transcriptions.txt"] + [f"outputs/{a}" for a in os.listdir("outputs/")]
output_csv = "combined_output.csv"

text_files_to_csv_with_headers(text_files, output_csv)

print("DONE!")