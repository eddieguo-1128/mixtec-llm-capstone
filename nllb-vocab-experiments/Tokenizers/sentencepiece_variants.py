import sentencepiece as spm

models = ['bpe', 'char', 'unigram']

model_name = 'unigram'
input_file = 'cleaned_transcriptions.txt'
vocab_size = 2000
output_file = f"outputs/{model_name}_{vocab_size}.txt"

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

# Train SentencePiece Model
spm.SentencePieceTrainer.train(input=input_file, 
                               model_prefix=model_name, 
                               model_type=model_name,
                               vocab_size=vocab_size)

# Tokenize the inputs!
input_texts = read_file_as_list(input_file)

sp = spm.SentencePieceProcessor(model_file=f'{model_name}.model')
tokenized_texts = sp.encode(input_texts, out_type=str)

tokenized_texts = [' '.join(s) for s in tokenized_texts]

# Write to output file
write_list_to_file(tokenized_texts, output_file)