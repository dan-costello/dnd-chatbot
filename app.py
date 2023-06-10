import os
import openai
from dotenv import load_dotenv
from langchain.llms import OpenAI
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)


llm = OpenAI(openai_api_key=st.secrets["openai_key"])

load_dotenv()
llm = OpenAI(temperature=0.1, verbose=True)
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
from constants import CHROMA_SETTINGS

store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
)
# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="clean_original", description="full recap of DND Adventures", vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

st.title("🐲🔥 Oghma")
prompt = st.text_input("Input your prompt here")
if prompt:
    response = agent_executor.run(prompt)
    st.write(response)
    # With a streamlit expander
    with st.expander("Relevant Passage"):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt)
        # Write out the first
        st.write(search[0][0].page_content)
