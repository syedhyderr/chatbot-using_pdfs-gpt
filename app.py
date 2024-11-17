import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from htmlTemplates import css, user_template, bot_template


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split the extracted text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    """Store text chunks in a ChromaDB vector store."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(
        texts=text_chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"  # Specify persistence directory
    )
    return vectorstore


def get_conversation_chain(vectorstore):
    """Set up the conversational retrieval chain."""
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question):
    """Handle user input and generate a response."""
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history=response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):

     if i%2 == 0:
        st.write(user_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)
     else:
        st.write(bot_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)



def main():
    """Main function to run the Streamlit app."""
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= None

    st.header("Chat with multiple PDFs:")
    user_question = st.text_input("Ask a question ")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store with ChromaDB
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
