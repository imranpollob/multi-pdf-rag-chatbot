import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if 'pdf_docs' not in st.session_state:
    st.session_state.pdf_docs = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = {}

def get_pdf_text_with_metadata(pdf_docs):
    """Extract text from PDF documents with metadata for each page"""
    text_chunks_with_metadata = []
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        filename = pdf.name
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            
            # Create chunk with metadata
            chunk_with_metadata = {
                'text': text,
                'metadata': {
                    'source': filename,
                    'page': page_num + 1  # Page numbers start from 1
                }
            }
            text_chunks_with_metadata.append(chunk_with_metadata)
    
    return text_chunks_with_metadata

def split_text_with_metadata(text_chunks_with_metadata):
    """Split text chunks into smaller chunks while preserving metadata"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    all_chunks = []
    all_metadatas = []
    
    for chunk_data in text_chunks_with_metadata:
        text = chunk_data['text']
        original_metadata = chunk_data['metadata']
        
        # Split the text
        sub_texts = text_splitter.split_text(text)
        
        # Create metadata for each sub-chunk
        for sub_text in sub_texts:
            all_chunks.append(sub_text)
            # Make a copy of the original metadata to avoid reference issues
            chunk_metadata = original_metadata.copy()
            all_metadatas.append(chunk_metadata)
    
    return all_chunks, all_metadatas

def get_vectorstore(text_chunks_with_metadata):
    """Create vector store from text chunks with preserved metadata"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Split text while preserving metadata
    text_chunks, metadatas = split_text_with_metadata(text_chunks_with_metadata)
    
    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vectorstore

def process_uploaded_pdfs(uploaded_files):
    """Process uploaded PDFs and create vector store"""
    # Extract text with metadata
    text_chunks_with_metadata = []
    
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        # Extract text with metadata
        pdf_reader = PdfReader(temp_path)
        filename = uploaded_file.name
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            
            # Create chunk with metadata
            chunk_with_metadata = {
                'text': text,
                'metadata': {
                    'source': filename,
                    'page': page_num + 1  # Page numbers start from 1
                }
            }
            text_chunks_with_metadata.append(chunk_with_metadata)
        
        # Clean up temp file
        os.unlink(temp_path)
    
    # Create vector store
    vectorstore = get_vectorstore(text_chunks_with_metadata)
    
    # Extract file info for display
    file_info = {}
    for uploaded_file in uploaded_files:
        # Count pages by temporarily reading the file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        pdf_reader = PdfReader(temp_path)
        pages_count = list(range(len(pdf_reader.pages)))
        file_info[uploaded_file.name] = pages_count
        
        # Clean up temp file
        os.unlink(temp_path)
    
    return vectorstore, file_info

def get_conversation_chain(vectorstore):
    """Create conversation chain with RAG using older approach for LangChain 0.3.x"""
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3
    )
    
    # Custom prompt template for better source citation
    template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide the sources (PDF file names and page numbers) where you found the information.
    
    {context}
    
    Question: {question}
    
    Helpful Answer with Sources:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 4}  # Retrieve top 4 most similar chunks
        ),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    
    return qa

def handle_user_question(user_question):
    """Handle user question and generate response with source citations"""
    if st.session_state.vectorstore is not None:
        # Get the conversation chain
        conversation_chain = get_conversation_chain(st.session_state.vectorstore)
        
        # Get response
        response = conversation_chain({"query": user_question})
        
        # Extract sources from the response
        sources_info = []
        if "source_documents" in response:
            for doc in response["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                sources_info.append({"source": source, "page": page})
        
        # Format response with source citations
        result = response["result"]
        
        # Add sources to the result if available
        if sources_info:
            result += "\n\n**Sources:**\n"
            unique_sources = []
            for source_info in sources_info:
                source_str = f"- {source_info['source']} (Page {source_info['page']})"
                if source_str not in unique_sources:
                    unique_sources.append(source_str)
            
            for source_str in unique_sources:
                result += f"{source_str}\n"
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": result})
    else:
        st.warning("Please upload and process PDFs first.")

def handle_user_question_with_streaming(user_question):
    """Handle user question and generate response with streaming"""
    if st.session_state.vectorstore is not None:
        # Get the conversation chain
        conversation_chain = get_conversation_chain(st.session_state.vectorstore)
        
        # Get response
        response = conversation_chain({"query": user_question})
        
        # Extract sources from the response
        sources_info = []
        if "source_documents" in response:
            for doc in response["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                sources_info.append({"source": source, "page": page})
        
        # Format response with source citations
        result = response["result"]
        
        # Add sources to the result if available
        if sources_info:
            result += "\n\n**Sources:**\n"
            unique_sources = []
            for source_info in sources_info:
                source_str = f"- {source_info['source']} (Page {source_info['page']})"
                if source_str not in unique_sources:
                    unique_sources.append(source_str)
            
            for source_str in unique_sources:
                result += f"{source_str}\n"
        
        # Display response in chat
        with st.chat_message("assistant"):
            st.write(result)
            
            # Display sources
            if sources_info:
                with st.expander("Show Sources"):
                    unique_sources = []
                    for source_info in sources_info:
                        source_str = f"- {source_info['source']} (Page {source_info['page']})"
                        if source_str not in unique_sources:
                            unique_sources.append(source_str)
                    
                    for source_str in unique_sources:
                        st.write(source_str)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": result})
    else:
        st.warning("Please upload and process PDFs first.")

def get_document_summaries():
    """Generate summaries for each uploaded document"""
    if st.session_state.vectorstore is not None:
        # Get all documents
        all_docs = st.session_state.vectorstore._collection.get()
        
        # Group docs by source
        docs_by_source = {}
        if 'metadatas' in all_docs and 'documents' in all_docs:
            for i, doc_text in enumerate(all_docs['documents']):
                if i < len(all_docs['metadatas']):
                    source = all_docs['metadatas'][i].get('source', 'Unknown')
                    if source not in docs_by_source:
                        docs_by_source[source] = []
                    docs_by_source[source].append(doc_text)
        
        summaries = {}
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        
        for source, texts in docs_by_source.items():
            # Combine texts for summarization
            combined_text = " ".join(texts)[:4000]  # Limit to avoid token issues
            
            # Create summarization prompt
            summary_prompt = f"""
            Write a concise summary of the following document. Highlight the main topics, 
            key points, and important information.
            
            Document text: {combined_text}
            
            Summary:"""
            
            # Generate summary
            summary = llm.predict(summary_prompt)
            summaries[source] = summary
        
        return summaries
    return {}

def get_query_suggestions(user_input):
    """Generate query suggestions based on user input and document content"""
    if st.session_state.vectorstore is not None and user_input.strip():
        # Get relevant documents to understand context
        docs = st.session_state.vectorstore.similarity_search(user_input, k=2)
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
        # Create prompt for generating suggestions
        suggestions_prompt = f"""
        Based on the following query and document context, generate 3-5 related follow-up questions 
        or query suggestions that would help explore the topic further:
        
        Original query: {user_input}
        
        Document context: {docs[0].page_content[:1000] if docs else 'No context available'}
        
        Please provide the suggestions as a numbered list:"""
        
        suggestions_text = llm.predict(suggestions_prompt)
        return suggestions_text
    return ""

def main():
    st.set_page_config(page_title="Multi-PDF RAG Chatbot", page_icon=":books:")
    st.header("📚 Multi-PDF RAG Chatbot")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Upload your PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        if st.button("Process PDFs"):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    # Process uploaded files
                    vectorstore, file_info = process_uploaded_pdfs(uploaded_files)
                    
                    # Update session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.uploaded_files_info = file_info
                    
                    # Show success message
                    st.success(f"Processed {len(uploaded_files)} PDF(s) successfully!")
                    for file in uploaded_files:
                        st.write(f"- {file.name}")
            else:
                st.warning("Please upload at least one PDF file.")
        
        # Document summarization
        if st.session_state.vectorstore is not None:
            if st.button("Generate Document Summaries"):
                with st.spinner("Generating summaries..."):
                    summaries = get_document_summaries()
                    st.session_state.document_summaries = summaries
    
    # Display uploaded files info
    if st.session_state.uploaded_files_info:
        st.subheader("Uploaded Files")
        for filename, pages in st.session_state.uploaded_files_info.items():
            st.write(f"- {filename} (pages: {len(pages)})")
    
    # Document summaries section
    if hasattr(st.session_state, 'document_summaries') and st.session_state.document_summaries:
        with st.expander("Document Summaries", expanded=False):
            for source, summary in st.session_state.document_summaries.items():
                with st.container():
                    st.subheader(f"📄 {source}")
                    st.write(summary)
    
    # Chat interface
    st.subheader("Chat with your documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        handle_user_question_with_streaming(user_question)
        
        # Generate query suggestions after the response
        with st.spinner("Generating query suggestions..."):
            suggestions = get_query_suggestions(user_question)
            if suggestions:
                with st.expander("💡 Query Suggestions", expanded=True):
                    st.write(suggestions)
        
        # Rerun to update the chat display
        st.rerun()

if __name__ == '__main__':
    main()