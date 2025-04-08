import json
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_files=["./sample.json"])
docs = reader.load_data()
print(f"Loaded {len(docs)} docs")
