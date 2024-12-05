import json

# Open the input and output files
path = "/home/aaditd/3_Capstone/Data/unsupervised.json"
with open(path, "r", encoding="utf-8") as infile, open("cleaned_transcriptions_2.txt", "w", encoding="utf-8") as outfile:
    for line in infile:
        # Load each JSON entry
        entry = json.loads(line)
        
        # Extract the "cleaned_transcription" field
        cleaned_transcription = entry.get("cleaned_transcription", "")
        
        # Write to the output file
        outfile.write(cleaned_transcription + "\n")