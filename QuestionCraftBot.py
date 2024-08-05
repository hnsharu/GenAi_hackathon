import os
import dotenv
import json
dotenv.load_dotenv()

from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from bs4 import BeautifulSoup
from langchain_google_genai import GoogleGenerativeAI
from datetime import date
from langchain.agents import tool
import langchain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
import streamlit as st
import time

langchain.debug=True

generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 64,
  "response_mime_type": "text/plain",
}

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ['GOOGLE_API_KEY'],generation_config=generation_config)

html_page = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quiz Form</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            color: #333;
        }}
        .container {{
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            max-width: 600px;
            width: 100%;
            height: 100%;
        }}
        h1 {{
            margin-top: 0;
            color: #007BFF;
        }}
        .question {{
            margin-bottom: 20px;
        }}
        .question p {{
            font-weight: bold;
        }}
        .options input {{
            margin-right: 10px;
        }}
        .shortAnswer input {{
            margin-right: 10px;
        }}
        .options label {{
            margin-right: 20px;
        }}
        .submit-btn {{
            background-color: #007BFF;
            border: none;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }}
        .submit-btn:hover {{
            background-color: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Quiz Form</h1>
        {html_code}
    </div>
</body>
</html>
"""

close_page="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Completed</title>
</head>
<body>
    <center>
    <h1>Answers Submitted Successfully,Please ask bot for your marks!</h1>
    <p>this page will close shortly</p>
    </center>
    <script>
        // Close the tab after a short delay to ensure processing is done
        setTimeout(function() {
            window.close();
        }, 5000);
    </script>
</body>
</html>
"""

@tool
def answer_validator(txt:str)->str:
    """  
    The Answer validator tool is used to validate the user's answers. This function should be called when the user requests their score or marks.\ 
    The input is always an empty string, and the function returns the status of the answer validation along with the user's marks.
    """

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="""You are an expert evaluator. Your task is to validate a user's answers against the actual answers and provide scores. The user's answers and actual answers are given in JSON format, with the question number as the key and the answer as the value.\

        Here are the rules:\

        1.Each question has equal weight.\
        2.If the user's answer matches the actual answer, they receive 1 point.\
        3.If the user's answer does not match the actual answer, they receive 0 points.\
        4.For short-answers, capture the meaning of the sentence and validate according to it, if user is too close to answer or has spelling mistake provide 0.5 points.\  
        5.Provide the total score and a detailed explanation of validation for each question.\
        
        Format the output as json follows:\n\
        
        The JSON object should have the following keys:

        Total_marks: The marks scored by the user out of number of questions.
        Question_wise_marks: A list of JSON objects, each containing:
          - question: The question number.
          - user_answer: The answer provided by the user.
          - correct_answer: The correct answer.
          - explanation: How you defend your validation with respect to relevant line.
          - relevant_line: Relevant line from actual_answers if any.
          - metadata: metadata from actual_answers if any.

    """  )
        
    data = {
        "user_answers": os.environ["user_answers"],
        "actual_answers": st.session_state["answers"]
    }
     


    response=model.generate_content(str(data))
    json_data=response.text
    print(json_data)
    if "```json" in response.text:
        json_data=response.text[8:]
        json_data=json_data[:-3]
    json_data=json.loads(json_data)
    
    st.write(json_data['Question_wise_marks'])
    st.session_state['chat_history'].extend(
    [
        AIMessage(content=str(json_data['Question_wise_marks']))
    ]
    )


    return f"User marks are validated successfully. User marks is {json_data['Total_marks']}."
    

@tool
def generate_questions(params:str)->str:
  """  
  The generate_questions tool generates an HTML form for questions. The input for this function is a JSON object containing the keys user_instruction and use_relevant_context, with an optional topic key if the tool needs to use its own knowledge.\ 
  The user_instruction key provides specific instructions for generating questions, such as difficulty levels, the number of questions, and types of questions. The use_relevant_context key determines whether the questions should be generated from relevant context ("yes") or from the tool's own knowledge ("no"). If the use_relevant_context is "no", the topic parameter must be included.\ 
  The output of this function is the status of the question generation process."""


  with st.spinner("Generating questions..."):
    model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction= """Create an HTML form for a questionnaire with respect to provided relevant context or from your knowledge. if relevant context is no then use your own knowledge for provided topic else The questions should be generated only from the provided relevant context. You will also be provided with difficulty level, question type and number of questions.\
    Generated questions should be only of types that user has specified and aslo the difficulty levels should be implemented. it should have exact number of questions that user mentioned.
    The following specifications are:

    1. The form element should have an ID of "QuestionnaireForm" and an action attribute set to "/submit" with a POST method.
    2. Inside the form, create a div with an ID of "questions".
    3. Within the "questions" div, create separate divs, each representing a question. Each question div should have the following structure:
      - A paragraph element (<p>) containing the Difficulty level of the question.
      - An ID of "question" (no numbers in the ID).
      - A paragraph element (<p>) containing the question text with question number.
      - For multiple-choice questions:
        - A nested div with an ID of "mcq" (no numbers in the ID) that contains three radio button inputs. Each radio button input should have:
          -- An ID of "option" (no numbers in the ID).
          -- A name attribute unique to the question (e.g., "question1", "question2").
          -- A value attribute representing the answer option (e.g., "a", "b", "c").
          -- A corresponding label element with a for attribute matching the radio button's ID and containing the answer text.
      - For short-answer questions:
        - A nested div with an ID of "shortAnswer" (no numbers in the ID) that contains a single text input field. The text input should have:
          -- An ID of "shortAnswerInput" (no numbers in the ID).
          -- A name attribute unique to the question (e.g., "question6", "question7").
      - For coding questions:
        - A nested div with an ID of "coding" (no numbers in the ID) that contains a textarea element. The textarea should have:
        -- An ID of "codingInput" (no numbers in the ID).
        -- A name attribute unique to the question (e.g., "question8", "question9").

    4. Include a submit button at the end of the form with the text "Submit".
    5. Remember not to add numbers in the IDs.
    6. Include only the form element and its contents, without any <!DOCTYPE html>, <html>, or <body> tags.

    Output the result in a JSON format with two keys:

    - "form": containing the HTML form code as a string.
    - "answers": containing a JSON object where the keys are the name attributes of the correct option or text input, and the values are the correct answers along with extracted relevant line from the context which has answers to questions generated.

    output_example when use_relevant context is yes:
    {
      "form": "<form id='QuestionnaireForm' action='/submit' method='post'>...</form>",
      "answers": {
        "question1": "correct answers, relevant_line: [line extracted from the provided context], metadata:[metadata of relevent line, which includes page number, source of the relevant line]",
        "question2": "correct answers, relevant_line: [line extracted from the provided context], metadata:[metadata of relevent line, which includes page number, source of the relevant line]",
        "question3": "correct answers, relevant_line: [line extracted from the provided context], metadata:[metadata of relevent line, which includes page number, source of the relevant line]",
        "question4": "correct answers, relevant_line: [line extracted from the provided context], metadata:[metadata of relevent line, which includes page number, source of the relevant line]",
        "question5": "correct answers, relevant_line: [line extracted from the provided context], metadata:[metadata of relevent line, which includes page number, source of the relevant line]",
        ...
      }
    }

    output_example when use_relevant context is no:
    {
      "form": "<form id='QuestionnaireForm' action='/submit' method='post'>...</form>",
      "answers": {
        "question1": "correct answers, explaination of correct answer]",
        "question2": "correct answers, explaination of correct answer]",
        "question3": "correct answers, explaination of correct answer]",
        "question4": "correct answers, explaination of correct answer]",
        "question5": "correct answers, explaination of correct answer]",
        ...
      }
    }
    """
  )

    if "```json" in params:
        params=params[8:]
        params=params[:-3]
    params=json.loads(params)

    user_prompt = f"user_instruction: {params['user_instruction']}\n\n" \
                f"{'Relevant context: ' + st.session_state['relevant_context'] if str(params['use_relevant_context']).lower() == 'yes' else 'Topic: ' + params['topic']}"

    print(user_prompt)
    response=model.generate_content(user_prompt)
    print(response.text)
    json_data=response.text
    if "```json" in response.text:
        json_data=response.text[8:]
        json_data=json_data[:-3]
    json_data=json.loads(json_data)
    st.session_state["html_code"]=json_data['form']
    st.session_state["answers"]=json_data['answers']

  return "Questions are Generated  successfully! Please continue further with hosting webpage"
  

from flask import Flask,request, render_template_string
import threading
app = Flask(__name__)


@tool
def host_webpage(text:str)->str:
    """
   The host_webpage function returns a link to the webpage where the generated questions are hosted.\ 
   This function should be invoked only after the questions have been successfully generated. \
   The input for this function is always an empty string, and it outputs the status of hosting along with the link to the hosted webpage"""

    html_content = html_page.format(html_code=st.session_state["html_code"])

    @app.route('/')
    def display_html():
        return render_template_string(html_content)
    
    @app.route('/submit', methods=['POST'])
    def submit():
        user_dict={}
        # Extracting user choices
        user_data=request.form
        for item in user_data.items():
            user_dict[item[0]]=item[1]
        os.environ["user_answers"]=json.dumps(user_dict)


        return close_page


    def run_app():
        app.run(host='localhost', port=5000, debug=False)

    # Start Flask in a separate thread
    global thread
    thread = threading.Thread(target=run_app)
    thread.start()
    # app.run(host='localhost', port=5000, debug=False)


    # Return the link to the page
    return "The webpage hosted successfully at http://localhost:5000/"

@tool
def get_revelent_context(topic: str) -> str:
    """The get_revelent_context function is employed to obtain relevant context for generating questions.\ 
    It should be utilized after the user has chosen a specific topic for question generation.\ 
    The input should be the topic selected by the user,\ 
    and the output will indicate the status of context retrieval, ensuring that the questions are generated based on the provided context."""
    
    relevent_doc=st.session_state["vectordb"].similarity_search(topic,k=2)
    st.session_state["relevant_context"]="\n\n".join([str(doc) for doc in relevent_doc])

    return "The relevent context retrived successfuly!No need to retrive again Please continue further with asking difficulty level"


@tool
def web_search(topic:str) -> str:
    """ The web_search tool function is used for conducting web searches based on a user-specified topic.\
        This function is to be used when the user wishes to search the web for information on a particular topic.\ 
        The input must be the topic the user is interested in, and the function will return a status that reflects the process of searching and retrieving the relevant data"""

    tool = TavilySearchResults(k=2)
    results =tool.invoke({"query": topic})
    relevent_context=""
    docs=[]
    for res in results:
        loader = WebBaseLoader(res['url'])
        data = loader.load()
        doc=text_splitter.split_documents(data)
        docs.extend(doc)

    st.session_state["vectordb"]= Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )

    relevent_doc=st.session_state["vectordb"].similarity_search(topic,k=5)
    st.session_state["relevant_context"]="\n\n".join([str(doc) for doc in relevent_doc])
 

    return "The web search successfuly!No need to search again Please continue further with asking difficulty level"



@tool
def provide_UI_for_upload_document(text: str) -> str:
    """The provide_UI_for_upload_document function is designed to offer a user interface for uploading documents.\
        It should only be used when the user states, "I want to upload a document." The input for this function is always an empty string,\ 
        and its output is a status indicating whether the UI was provided or not."""

    st.write("please upload your document")
    st.file_uploader("Your file","",key="file_uploader")
    return "The users is all set to upload document now!"

tools=[provide_UI_for_upload_document,web_search,get_revelent_context,generate_questions,host_webpage,answer_validator]

agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,

    handle_parsing_errors=True,
    conversational=True,
    max_iterations=5,
    verbose = True)
 

template="""
Assistant is a large language model trained on vast data

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

You are an user-friendly Question Generating Assistant. Answer the human message accordingly and orchestrate the process of generating questions (MCQs/ Short Answers/coding) based on workflow defined and tool outputs as best you can. 
The assistant only responds to queries related to question generation as defined in the application description.
you need to orchestrate the flow correctly as defined below
always follow these steps:

1. Greet the user and ask for the source of the content for the questions (Upload a document, Use API to search the web, Use LLM's own knowledge).

2. Based on the chosen source, follow these steps:

    - If "Upload a document" is chosen:
        - Use the `provide_ui_to_upload` tool so the user can upload their document.
        - Once the document is loaded, ask for the topic.
        - Use the `get_relevant_context` tool to extract the relevant context for the topic.
        - Proceed to Step 3.

    - If "Use API to search the web" is chosen:
        - Ask the user for the topic they want to search for on the web.
        - Use the `web_search` tool with the user-mentioned topic.
        - Proceed to Step 3.

    - If "Use LLM's own knowledge" is chosen:
        - Ask for the topic.
        - Proceed to Step 3.

3. Collect user details for question generation:
    - Ask the user for the following details:
        - Difficulty level of the questions (Easy, Medium, Hard)
        - Question type (Multiple-choice questions, short-answers, coding questions)
        - Number of questions (greater than 0 and less than 11)
    - Extract the difficulty level, question type, and number of questions from the user input.
    - Ask the user for any missing details if they were not provided.
    - Ensure you have all three details and confirm user provided details with user before proceeding.

4. Generate questions:
    - If the source is "Upload a document" or "Use API to search the web," use the `generate_question` tool with the `user_instruction` parameter, including the specific details provided by the user, and set the `use_relevant_context` parameter to 'yes'. Format the request in JSON.
    - If the source is "Use LLM's own knowledge," use the `generate_question` tool with the `user_instruction` parameter, including the specific details provided by the user, and set the `use_relevant_context` parameter to 'no'. Additionally, pass the `topic` parameter with the user-specified topic. Format the request in JSON.

5. Use the `host_webpage` tool to host the generated questions and provide the link to the user.

6. Use the `validate_answer` tool when the user asks for marks or scores. Pass the output from the answer validator tool directly, without any modifications or additional tool calls.
TOOLS:
------

Assistant has access to the following tools:

> provide_UI_for_upload_document: The provide_UI_for_upload_document function is designed to offer a user interface for uploading documents. It should only be used when the user states, "I want to upload a document." The input for this function is always an empty string,\
        and its output is a status indicating whether the UI was provided or not.        
> web_search: The web_search tool function is used for conducting web searches based on a user-specified topic. This function is to be used when the user wishes to search the web for information on a particular topic.\
       The input must be the topic the user is interested in, and the function will return a status that reflects the process of searching and retrieving the relevant data       
> get_revelent_context: The get_revelent_context function is employed to obtain relevant context for generating questions.\
    It should be utilized after the user has chosen a specific topic for question generation.\
    The input should be the topic selected by the user,\
    and the output will indicate the status of context retrieval, ensuring that the questions are generated based on the provided context.
> generate_questions: The generate_questions tool generates an HTML form for questions. The input for this function is a JSON object containing the keys user_instruction and use_relevant_context, with an optional topic key if the tool needs to use its own knowledge.\
    The user_instruction key provides specific instructions for generating questions, such as difficulty levels, the number of questions, and types of questions. The use_relevant_context key determines whether the questions should be generated from relevant context ("yes") or from the tool's own knowledge ("no"). If the use_relevant_context is "no", the topic parameter must be included.\
    The output of this function is the status of the question generation process.
> host_webpage: The host_webpage function returns a link to the webpage where the generated questions are hosted.\
    This function should be invoked only after the questions have been successfully generated. The input for this function is always an empty string, and it outputs the status of hosting along with the link to the hosted webpage
> answer_validator: The Answer validator tool is used to validate the user's answers. This function should be called when the user requests their score or marks.\
    The input is always an empty string, and the function returns the status of the answer validation along with the user's marks.
    
When you have to use a tool, You MUST use the tool in below format:


Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [provide_UI_for_upload_document,web_search, get_revelent_context, generate_questions, host_webpage, answer_validator]
Action Input: the input to the action
Observation: the result of the action.


You are allowed to do Single Thought and Response at a time, Don't do multiple Thought at a time.
Do not include any extra explanatory text outside this format when using tool. Only use the specified format strictly.


When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the below format:


Thought: Do I need to use a tool? No
AI: [your response here].


Whenver User ask any information outside of the Scope of the Application , please return saying "Sorry , I could not help you with this Request , Do you want to upload a document, search the web for information, or use my own knowledge to generate questions?"
Do not include any extra explanatory text outside this format. Only use the specified format strictly.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

# print(agent.agent.llm_chain.prompt.template)
agent.agent.llm_chain.prompt.template=template

st.set_page_config(page_title="Question Craft Bot", page_icon="🧠")


# Custom CSS for Streamlit app
st.markdown("""
    <style>
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border: 1px solid #007BFF;
        border-radius: 5px;
    }
    .stMarkdown {
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #007BFF;'>Question Craft Bot</h1>", unsafe_allow_html=True)


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "regenerate" in st.session_state:

    if st.session_state['regenerate']:
        if 'file_uploader' in st.session_state:
            st.session_state.pop('file_uploader')
        prompt=None
        while not isinstance(prompt,HumanMessage):
            prompt=st.session_state["chat_history"].pop()

        for message in st.session_state["chat_history"]:
            if isinstance(message,HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:
                with st.chat_message("Assistance"):
                    st.markdown(message.content)
        
        with st.chat_message("user"):
            st.markdown(prompt.content)

        with st.chat_message("assistant"):
            response=agent({"input": prompt, "chat_history": st.session_state["chat_history"]})
            st.markdown(response["output"])

        st.session_state['chat_history'].extend(
        [
            prompt,
            AIMessage(content=response["output"])
        ]
        )
    else:
        for message in st.session_state["chat_history"]:
            if isinstance(message,HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:
                with st.chat_message("Assistance"):
                    st.markdown(message.content)
print(st.session_state)







if "file_uploader" in st.session_state and st.session_state["file_uploader"]!="" and st.session_state["file_uploader"]!=None:
    with st.spinner("Loading Document..."):
        print(st.session_state["file_uploader"]!=None)
        current_directory = os.getcwd()
        with open(os.path.join(current_directory,st.session_state["file_uploader"].name),"wb") as f:
            f.write(st.session_state["file_uploader"].getbuffer())
        path = os.path.join(current_directory,st.session_state["file_uploader"].name)

        loader = PyPDFLoader(path)
        data = loader.load()
        docs=text_splitter.split_documents(data)
        st.session_state["vectordb"]= Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
        )
        # st.write("Successfully Loaded")

        with st.chat_message("assistant"):
            response=agent({"input": "User Document uploaded successfully! please continue with asking topic", "chat_history": st.session_state["chat_history"]})
            st.markdown(response["output"])
            
        st.session_state['chat_history'].extend(
        [
            AIMessage(content=response["output"])
        ]
        )
        
        st.session_state.pop('file_uploader')

if prompt := st.chat_input("type here ..."):
    st.session_state['chat_history'].extend(
    [
        HumanMessage(content=prompt)
    ]
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response=agent({"input": prompt, "chat_history": st.session_state["chat_history"]})
        st.markdown(response["output"])
    st.session_state['chat_history'].extend(
    [
        AIMessage(content=response["output"])
    ]
    )


if st.session_state['chat_history']!=[]:
    st.button("🔁",key="regenerate")

print(st.session_state)