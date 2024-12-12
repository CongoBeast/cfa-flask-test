from flask import Flask, request, jsonify
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up Flask app
app = Flask(__name__)

# Set up the OpenAI language model
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0,
    max_tokens=1000,
    openai_api_key=openai_api_key
)

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def openai_insights(data_story, llm=llm, top_k=5, max_context_tokens=6000):
    pdf_text = extract_text_from_pdf("./DSEC02.pdf")
    pdf_document = Document(metadata={'source': 'pdf_data'}, page_content=pdf_text)

    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_documents = text_splitter.split_documents([pdf_document])

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_documents, embedding_model)

    prompt_template = """
    You are an AI tutor specializing in CFA exam preparation. Your role is to help students understand CFA concepts, practice problem-solving, and retain key information.
    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.get_relevant_documents(data_story)

    total_tokens = sum(len(doc.page_content.split()) for doc in relevant_docs)

    if total_tokens > max_context_tokens:
        summarized_context = []
        for doc in relevant_docs:
            summary_prompt = f"Summarize the following text for a CFA exam study guide:\n\n{doc.page_content}"
            summarized_chunk = llm.predict(summary_prompt)
            summarized_context.append(summarized_chunk["text"])
        context = "\n".join(summarized_context)
    else:
        context = "\n".join([doc.page_content for doc in relevant_docs])

    response = qa_chain.run(input_documents=[Document(page_content=context)], question=data_story)

    return response

@app.route('/insights', methods=['POST'])
def insights():
    if request.method == 'POST':
        # Get the question from the request
        data = request.json
        if 'question' not in data:
            return jsonify({"error": "Missing 'question' field in request"}), 400

        question = data['question']

        try:
            # Get the insights using the provided function
            response = openai_insights(question)
            return jsonify({"response": response}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        print(name)
        return f"Hello, {name}!"
    else:
        return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
