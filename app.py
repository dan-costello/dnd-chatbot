from langchain.llms import OpenAI
import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from constants import CHROMA_SETTINGS

# Intialize environmental variables
os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
embeddings_model_name = st.secrets["EMBEDDINGS_MODEL_NAME"]
persist_directory = st.secrets["PERSIST_DIRECTORY"]

# Intialize LLM, vector database and LangChain agent.
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
)
vectorstore_info = VectorStoreInfo(
    name="clean_original", description="full recap of DND Adventures", vectorstore=store
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

# Create streamlit app
st.title("🐲👁️ BB+B Beholder")
prompt = st.text_input("What question do you ponder?")
if prompt:
    response = agent_executor.run(prompt)
    st.write(response)
    # With a streamlit expander
    with st.expander("Relevant Passage"):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt)
        # Write out the first
        st.write(search[0][0].page_content)
