# Medical-Graph-RAG
We build a Graph RAG System specifically for the medical domain.

Check our paper here: https://arxiv.org/html/2408.04187v1

# Quick Start
1. conda env create -f medgraphrag.yml

2. export OPENAI_API_KEY = your OPENAI_API_KEY

3. python run.py -simple True (now using ./dataset_ex/report_0.txt as RAG doc, "What is the main symptom of the patient?" as the prompt, change the prompt in run.py as you like.)

# Build from scratch
## Prepare Neo4j and LLM: 
prepare neo4j and LLM (using ChatGPT here for an example), you need to export:

export OPENAI_API_KEY = your OPENAI_API_KEY

export NEO4J_URL= your NEO4J_URL

export NEO4J_USERNAME= your NEO4J_USERNAME

export NEO4J_PASSWORD= your NEO4J_PASSWORD

## Construct the graph (use "mimic_ex" dataset as an example): 
1. Download mimic_ex here, put that under your data path, like ./dataset/mimic_ex

2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -grained_chunk True -ingraphmerge True -construct_graph True

## Model Inference: 
1. put your prompt to ./prompt.txt

2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -inference True