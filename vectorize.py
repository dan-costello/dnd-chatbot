import os
from dotenv import load_dotenv
from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import glob
from langchain.document_loaders import TextLoader
from constants import CHROMA_SETTINGS
from multiprocessing import Pool
from tqdm import tqdm
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
)

load_dotenv()

# Load environment variables
persist_directory = os.environ.get("PERSIST_DIRECTORY")
source_directory = os.environ.get("SOURCE_DIRECTORY", "source_documents")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
chunk_size = 500
chunk_overlap = 50

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for i, doc in enumerate(
                pool.imap_unordered(load_single_document, filtered_files)
            ):
                results.append(doc)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    # save as text file
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    # save texts to files with name file+number
    count = 0

    for i, text in enumerate(texts):
        if i < 200:
            with open(f"texts/file{i}.txt", "w", encoding="utf-8") as f:
                f.write(text.page_content)
                count += 1
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, "index")):
        if os.path.exists(
            os.path.join(persist_directory, "chroma-collections.parquet")
        ) and os.path.exists(
            os.path.join(persist_directory, "chroma-embeddings.parquet")
        ):
            list_index_files = glob.glob(os.path.join(persist_directory, "index/*.bin"))
            list_index_files += glob.glob(
                os.path.join(persist_directory, "index/*.pkl")
            )
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def main():
    embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
    persist_directory = os.environ.get("PERSIST_DIRECTORY")

    model_type = os.environ.get("MODEL_TYPE")
    model_path = os.environ.get("MODEL_PATH")
    model_n_ctx = os.environ.get("MODEL_N_CTX")
    target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))

    from constants import CHROMA_SETTINGS

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        collection = db.get()
        texts = process_documents(
            [metadata["source"] for metadata in collection["metadatas"]]
        )
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=persist_directory,
            client_settings=CHROMA_SETTINGS,
        )
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
