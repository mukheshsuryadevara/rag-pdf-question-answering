from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def ask_question(vector_store, question):
    llm = ChatOllama(
        model="llama3",   
        temperature=0.2
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:"""
    )

    formatted_prompt = prompt.format(context=context, question=question)

    response = llm.invoke(formatted_prompt)

    return response.content


def main(pdf_path, question):
    print("Loading PDF...")
    text = load_pdf_text(pdf_path)

    print("Chunking...")
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")

    print("Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("Asking question...")
    answer = ask_question(vector_store, question)

    return answer


if __name__ == "__main__":
    pdf_path = "sample.pdf"

    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = main(pdf_path, question)

        print("\n====================")
        print("Answer:")
        print(answer)
        print("====================")

