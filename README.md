# Medical-Graph-RAG
We build a Graph RAG System specifically for the medical domain.

Check our paper here: https://arxiv.org/abs/2408.04187

## Demo
a docker demo is here: https://hub.docker.com/repository/docker/jundewu/medrag-post/general
 
Use it by: docker run -it --rm --storage-opt size=10G -p 7860:7860 \ -e OPENAI_API_KEY= your_key -e NCBI_API_KEY= your_key medrag-post

this demo used web-based searches on PubMed instead of locally storing medical papers and textbooks to detour the license wall.

## Quick Start (Baseline: a simple Graph RAG pipeline on medical data)
1. conda env create -f medgraphrag.yml

2. export OPENAI_API_KEY = your OPENAI_API_KEY

3. python run.py -simple True (now using ./dataset_ex/report_0.txt as RAG doc, "What is the main symptom of the patient?" as the prompt, change the prompt in run.py as you like.)

## Build from scratch (Complete Graph RAG flow in the paper)

### About the dataset
#### Paper Datasets
**Top-level Private data (user-provided)**: we used [MIMIC IV dataset](https://physionet.org/content/mimiciv/3.0/) as the private data.

**Medium-level Books and Papers**: We used MedC-K as the medium-level data. The dataset sources from [S2ORC](https://github.com/allenai/s2orc). Only those papers with PubMed IDs are deemed as medical-related and used during pretraining. The book is listed in this repo as [MedicalBook.xlsx](https://github.com/MedicineToken/Medical-Graph-RAG/blob/main/MedicalBook.xlsx), due to licenses, we cannot release raw content. For reproducing, pls buy and process the books.

**Bottom-level Dictionary data**: We used [Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/index.html) as the bottom level data. To access it, you'll need to create an account and apply for usage. It is free and approval is typically fast.

In the code, we use the 'trinity' argument to enable the hierarchy graph linking function. If set to True, you must also provide a 'gid' (graph ID) to specify which graphs the top-level should link to. UMLS is largely structured as a graph, so minimal effort is required to construct it. However, MedC-K must be constructed as graph data. There are several methods you can use, such as the approach we used to process the top-level in this repo (open-source LLMs are recommended to keep costs down), or you can opt for non-learning-based graph construction algorithms (faster, cheaper, and generally noisier)

#### Example Datasets
Recognizing that accessing and processing all the data mentioned may be challenging, we are working to provide simpler example dataset to demonstrate functionality. Currently, we are using the mimic_ex [here](https://huggingface.co/datasets/Morson/mimic_ex) here as the Top-level data, which is the processed smaller dataset derived from MIMIC. For Medium-level and Bottom-level data, we are in the process of identifying suitable alternatives to simplify the implementation, welcome for any recommendations.

### 1. Prepare the environment, Neo4j and LLM
1. conda env create -f medgraphrag.yml


2. prepare neo4j and LLM (using ChatGPT here for an example), you need to export:

export OPENAI_API_KEY = your OPENAI_API_KEY

export NEO4J_URL= your NEO4J_URL

export NEO4J_USERNAME= your NEO4J_USERNAME

export NEO4J_PASSWORD= your NEO4J_PASSWORD

#### Using MiniMax as an alternative LLM provider

Medical-Graph-RAG supports [MiniMax](https://www.minimaxi.com/) as an alternative LLM provider via their OpenAI-compatible API. MiniMax-M2.7 offers 204K context window, which is well-suited for processing long medical documents.

To use MiniMax instead of OpenAI:

```bash
export MINIMAX_API_KEY=your_minimax_api_key
# Optional: explicitly set the provider (auto-detected from MINIMAX_API_KEY)
export LLM_PROVIDER=minimax
# Optional: override the default model
export LLM_MODEL=MiniMax-M2.7-highspeed
```

Available MiniMax models: `MiniMax-M2.7` (default, 204K context), `MiniMax-M2.7-highspeed` (faster, 204K context).

> **Note:** MiniMax does not provide a public embedding API. When using MiniMax as the LLM provider, embeddings will still use OpenAI's `text-embedding-3-small` model. Ensure `OPENAI_API_KEY` is set for embedding operations.

For the `nano_graphrag` pipeline, pass the MiniMax completion function:

```python
from nano_graphrag import GraphRAG
from nano_graphrag._llm import minimax_m27_complete

graph_func = GraphRAG(
    working_dir="./nanotest",
    best_model_func=minimax_m27_complete,
    cheap_model_func=minimax_m27_complete,
)
```

For the CAMEL agent framework, use the MiniMax model type:

```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.MINIMAX,
    model_type=ModelType.MINIMAX_M27,
    model_config_dict={"temperature": 0.2},
)
```

### 2. Construct the graph (use "mimic_ex" dataset as an example)
1. Download mimic_ex [here](https://huggingface.co/datasets/Morson/mimic_ex), put that under your data path, like ./dataset/mimic_ex

2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -grained_chunk -ingraphmerge -construct_graph

### 3. Model Inference
1. put your prompt to ./prompt.txt

2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -inference

## Acknowledgement
We are building on [CAMEL](https://github.com/camel-ai/camel), an awesome framework for construcing multi-agent pipeline.

## Cite
~~~
@article{wu2024medical,
  title={Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation},
  author={Wu, Junde and Zhu, Jiayuan and Qi, Yunli},
  journal={arXiv preprint arXiv:2408.04187},
  year={2024}
}
~~~
