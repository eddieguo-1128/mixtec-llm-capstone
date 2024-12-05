import re

text = "nu14u(3)=un4"
input_file = 'cleaned_transcriptions.txt'
output_file = f"outputs/human.txt"

# Remove hyphens if needed, while retaining all other punctuation as specified
result = re.sub(r'-', '', re.sub(r'=', ' ', text))

def remove_newline_and_spaces(string):
  return string.replace("\n", "").replace(" ", "")

def read_file_as_list(filename):
    """Reads a text file and returns its content as a list of lines."""

    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines

def write_list_to_file(list_to_write, filename):
    """Writes a list to a text file, one item per line.

    Args:
        list_to_write (list): The list to be written to the file.
        filename (str): The name of the file to write to.
    """

    with open(filename, 'w') as f:
        for item in list_to_write:
            f.write(str(item) + '\n')


input_texts = read_file_as_list(input_file)

# input_texts = ["aapple=b", "a=b=c", "d-f"]

human_tokenized_text = [re.sub(r'-', '', re.sub(r'=', ' ', remove_newline_and_spaces(t))) for t in input_texts]


write_list_to_file(human_tokenized_text, output_file)