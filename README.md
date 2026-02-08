Moriarty: Star Trek RAG Setup Guide

Moriarty is a RAG LLM intended to serve as an expert on Star Trek lore, providing expert knowledge with a fair amount of condescension and arrogance. He's wrong a lot. Work in progress.

This document outlines the complete environment and dependency list, as well as the instructions to run the Memory Alpha XML ingestion script and the Moriarty Query script.

1. System Requirements (Fedora Linux)

Run the following to install the necessary compilers and system libraries required for lxml and chromadb to build correctly:

sudo dnf update
sudo dnf install -y python3-devel gcc gcc-c++ libxml2-devel libxslt-devel sqlite-devel

2. RAG Data
	
	I downloaded the current database from "https://memory-alpha.fandom.com/wiki/Memory_Alpha:Database_download" and placed it in ./data/ renamed as memoryalpha6feb.xml. If you change the name you will need to modify database_chunker.py.

3. LLM Engine (Ollama)

The system uses Ollama as the local backend for Llama 3.1.

    Install Ollama:

	    Bash command:

	    curl -fsSL https://ollama.com/install.sh | sh

    Pull the Model:

	    Bash command:

	    ollama pull llama3.1

4. Python Dependencies

Install all required Python libraries using pip.

	Bash command:

pip install chromadb \
            tiktoken \
            lxml \
            nltk \
            pyyaml \
            requests \
            onnxruntime \
            sentence-transformers

Dependency Breakdown:

	    chromadb: Vector database for storage.

	    tiktoken: Token counting for Llama 3.1 context windows.
	
	    lxml: High-speed, memory-efficient XML streaming.

	    nltk: Sentence-level splitting.

	    pyyaml: Configuration management.

	    onnxruntime: Local execution of the embedding model.

5. Project Structure


```text
moriarty
├── config.yaml                                  # config file
├── data                                         # folder for ingestion - create this folder and place your downloaded XML file here
│   └── memoryalpha6feb.xml                      # Memory Alpha XML dump
├── database_chunker.py                          # tokenizes, Chunks, Vectorizes
├── moriarty_query.py                            # query script
├── moriarty.sh                                  # initialization script
├── processed_chunks.json                        # chunked output - you won't have this until you run the ingesting script
├── README.md                                    # readme
└── star_trek_db                                 # Vectorized database - you will not have this until running database_chunker.py
    ├── 0c20eab4-e26a-49a1-99c5-c4479db16800
    │   ├── data_level0.bin
    │   ├── header.bin
    │   ├── index_metadata.pickle
    │   ├── length.bin
    │   └── link_lists.bin
    └── chroma.sqlite3

6. Tokenizing, Chunking, Vectorizing

	Run database_chunker.py referencing the XML file to ingest. Example command: python database_chunker.py ./data/memoryalpha6feb.xml

7. Activation

	Executing moriarty.sh will automatically launch local Ollama API and run program.
