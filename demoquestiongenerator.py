import os
import dotenv
dotenv.load_dotenv()

# import google.generativeai as genai

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = GoogleGenerativeAI(model="gemini-1.5-flash")

#loading data
def load_data(path):
    print("Loading the data.....")
    loader = PyPDFLoader(path)
    data = loader.load()
    docs=text_splitter.split_documents(data)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory='docs/chroma/'
    )
    return vectordb

def generate_questions():
    path=input("""
                Please provide the path to document. to use default document press 0.
            """)
    if path=="0":
        vectordb=load_data("assets/AWSCertifiedDataEngineerSlides.pdf")
    else:
        vectordb=load_data(path)
    

    while True:

        topic = input("""
                    Please provide the topic to generate questions. topics for default document are aws services. 
                    press 0 to exit.
                """)
        
        if topic=="0":
            break

        # Define the inner PromptTemplate
        chain_prompt = PromptTemplate(
            input_variables=['context', 'question'],
            template=(
                 """You are an assistant to generate 5 Multiple choice questions and answers in the format
                    Question no.
                    ...question here...

                    option:
                     ...options here...
                    .
                    Use the following pieces of retrieved context to create the question. 
                    topic to create question is provided in "Topic".
                    If you don't know the related context to create questions, just say that you don't know.
                    Topic: {question}
                    Context: {context}

                    Answer:"""
            )
        )
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=chain_prompt
        )

        # Create the ChatPromptTemplate
        prompt = ChatPromptTemplate(
            input_variables=['context', 'question'],
            messages=[human_message_prompt]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(
                search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2}
            ),
            verbose=True,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        result = qa_chain({"query":topic},)
        print(result["result"])


generate_questions()