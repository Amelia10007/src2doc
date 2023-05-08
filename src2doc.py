import os

import faiss

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory

############### How to use this ###############
# create a docker container from a python3 image
# pip install openai langchain faiss-cpu
# set env OPENAI_API_KEY to your OpenAI API key
#
# reference:
# https://python.langchain.com/en/latest/modules/memory/types/vectorstore_retriever_memory.html
############### How to use this ###############

############### Configuration ###############
REPOSITOERY_DIR = "./repository/leptrino-force-torque-sensor-rs"

QUESTIONS = [
    "What is the key functionality of this repository?",
    "What the function parse_command() does?",
    "What the function zeroed() does?"
]

LOG_FILE = "./qa.txt"
############### Configuration ###############



############### Main Routine ###############
print("Process: Load your repository...")

docs = []
for dirpath, dirnames, filenames in os.walk(REPOSITOERY_DIR):
    for file in filenames:
        path = os.path.join(dirpath, file)
        # Exclude git files in the target repository
        if ".git" in path:
            print(f"Warn: skip loading {path}")
            continue

        try:
            loader = TextLoader(path, encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            print(f"Warn: failed to load file {path}: {e}")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)



print("Process: Setting up a vector store...")

# This step varies what vectore store you use
embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

search_kwargs = {
    'distance_metric': 'cos',
    'fetch_k': 20,
    'k': 10,
    'maximal_marginal_relevance': True
}
retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
memory = VectorStoreRetrieverMemory(retriever=retriever)

for text in texts:
    source_filename = text.metadata['source']
    inputs = {"input": source_filename}
    outputs = {"output": text.page_content}
    memory.save_context(inputs=inputs, outputs=outputs)



print("Process: Setting up LLM...")

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0) # T=0 means deterministic behavior
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, verbose=True)
chat_history = []



print("Process: execute Q/A...")

for question in QUESTIONS:
    entities = {
        "question": question,
        "chat_history": chat_history
    }
    answer = qa.run(entities)
    chat_history.append((question, answer))



print(f"Process: save Q/A to {LOG_FILE}...")

with open(LOG_FILE, "w") as f:
    for question, answer in chat_history:
        f.write(f"Question:\n{question}\n")
        f.write(f"Answer:\n{answer}\n\n")
