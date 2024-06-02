from flask import Flask,request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os
from datetime import datetime, time
from flask_cors import CORS

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    chunks = text_splitter.split_text   (text)
    return chunks

def get_vectors_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context,make sure to provide all the details , if the answer is not present in the context then answer it as "Not present in the context", dont provide wrong answers.

    context: {context}
    question: {question}
    answer:  
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt= PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff",prompt=prompt)
    return chain

@tool
def user_input(user_question:str)->str:
    """ 
    this functions helps to answer the questions about baratie restaurant and its staff
    anything other than booking a table can be asked here
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs,"question":user_question}
        , return_only_outputs=True)
    return response
tables_baratie= {
    '7' : 4,
    '8' : 4,
    '9' : 4,
    '10' : 4,
    '11' : 4,
    '12' : 4,
    '13' : 4,
    '14' : 4,
    '15' : 4,
    '16' : 4,
    '17' : 4,
    '18' : 4,
    '19' : 4,
    '20' : 4,
    '21' : 4,
    '22' : 4,
    '23' : 4
}
@tool
def book_table(hour :str)-> str:
    """
    this function helps to book a table/appointment in the restaurant, baratie at a specific time

    args:
    hour: str : the hour at which the table is to be booked or it can be 'now' to book the table at the current hour. 

    """
    hr = hour.lower()
    hr = str(hr)
    if hr == 'now':
        now = datetime.now().time()
    else:
        if(":" in hr):
            hr = hr.split(":")[0]
        elif('pm' in hr):
            hr = int(hr.split("pm")[0])
            if(hr < 12):
                hr += 12
        elif('am' in hr):
            hr = int(hr.split("am")[0])
        now = time(int(hr), 0, 0)
    start_time = time(7, 0, 0)  
    end_time = time(23, 0, 0)
    if now < start_time or now > end_time:
        return {'output_text': "Sorry, we are closed at {now}. Please visit us between 7:00 AM to 9:00 PM"}
    # 14 slots each slot 4 tables
    current_hour = now.hour
    if(tables_baratie[str(current_hour)] == 0):
        return {'output_text': "Sorry, all tables are booked for this hour. Please choose another hour"}
    tables_baratie[str(current_hour)] -= 1
    # print(tables_baratie)
    return {'output_text': f"Table booked successfully at {current_hour} for 1 hour. Enjoy your meal!"}

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

raw_text = read_text_file("baratie.txt")
text_chunks = get_text_chunks(raw_text)
get_vectors_store(text_chunks)

tools=[user_input,book_table]
llm_with_tools = llm.bind_tools(tools)

app = Flask(__name__)
CORS(app,resources={r"/*": {"origins": "*"}})

@app.route('/',methods=['GET','POST'])
def hello_world():
    if(request.method == "POST"):
        messages = []
        data = request.get_json()
        user_question = data['data']['question']
        messages.append({'msg_type':'user','user_question':user_question})
        ai_msg= llm_with_tools.invoke(user_question)
        # print(ai_msg)
        for tool_call in ai_msg.tool_calls:
            selected_tool = {"user_input": user_input, "book_table": book_table}[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            # print(tool_output)
            # print(tool_output['output_text'])
            messages.append({'msg_type':'tool','tool_output':tool_output['output_text']})
        return {"messages":messages,"available_tables":tables_baratie}
    return {
        "message": "welcome to Baratie! We are a restaurant that serves delicious food!",
        "available_tables": tables_baratie
    }

if __name__ == "__main__":
    app.run(use_reloader=True,debug=True)