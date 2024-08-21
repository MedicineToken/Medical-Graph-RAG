
import os

def load_high(datapath):
    all_content = ""  # Initialize an empty string to hold all the content
    with open(datapath, 'r', encoding='utf-8') as file:
        for line in file:
            all_content += line.strip() + "\n"  # Append each line to the string, add newline character if needed
    return all_content





