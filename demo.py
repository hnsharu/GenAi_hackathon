import os
import dotenv
import json
dotenv.load_dotenv()

from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain_google_genai import GoogleGenerativeAI
from datetime import date
from langchain.agents import tool
import langchain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
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
            height: 100vh;
            color: #333;
        }}
        .container {{
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            max-width: 600px;
            width: 100%;
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

@tool
def answer_validator(txt:str)->str:
    """  
    The answer_validator tool validates the user's answers against the actual answers and provides the detailed explaination validation process with scores.
    Input should be a empty string. use this tool when user asks for the score or marks,\
    This tool returns status of validation and the details of validation process and users score.\
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
        
        Format the output as follows:\n\
        
        Total marks: [total marks here]\n\n\
        Question wise marks:\
            Question [number]: [1 or 0 points]\
            User's answer: [user's answer]\
            Correct answer: [correct answer]\
            Explanation: [why the user's answer is incorrect, if applicable]\
            ....
        
        Example Output:

        Total marks: 2 \n\n\
        Question wise marks:\n\

            Question 1: 1 point\n\
                User's answer: a\n\
                Correct answer: a\n\n\
            Question 2: 0 points\n\
                User's answer: b\n\
                Correct answer: c\n\n\
                Explanation: Option b is incorrect because [explanation]. The correct answer is c.\
              """  )
        

    data = {
        "user_answers": os.environ["user_answers"],
        "actual_answers": st.session_state["answers"]
    }
     


    response=model.generate_content(str(data))


    return "User Answers Validated Successfully \n\n"+ response.text
    

@tool
def generate_questions(params:str)->str:
  """  
  The generate_question tool generates an HTML form for a questions.\
  input should the json with four keys (difficulty,qtype,number_of_questions,use_relevant_context) plus one optional key (topic) when llm has to use its own knowledge. difficulty: level of the questions to be generated, values allowed are (easy, medium, hard), qtype: type of the questions to be generated, it should be one of (Multiple-choice-questions, short-answers) and number_of_questions: how many questions user want to be generated \
  use_relevant_context: yes, if question should be generated from relevant context, no, if llm have to use its own knowledge to generate questions. if use_relevent_context is no then topic parameter should be there
  return the status of question generated.\
  """

  model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction= """Create an HTML form for a questionnaire with respect to provided relevant context or from your knowledge. if relevant context is no then use your own knowledge for provided topic else The questions should be generated only from the provided relevant context. You will also be provided with difficulty level, question type and number of questions.\
    Generated questions should be only of type that user has specified and aslo the difficulty level should be implemented. it should have exact number of question that user mentioned.
    The following specifications are:

    1. The form element should have an ID of "QuestionnaireForm" and an action attribute set to "/submit" with a POST method.
    2. Inside the form, create a div with an ID of "questions".
    3. Within the "questions" div, create five separate divs, each representing a question. Each question div should have the following structure:
      - An ID of "question" (no numbers in the ID).
      - A paragraph element (<p>) containing the question text with question number.
      - For multiple-choice questions:
        - A nested div with an ID of "options" (no numbers in the ID) that contains three radio button inputs. Each radio button input should have:
          -- An ID of "option" (no numbers in the ID).
          -- A name attribute unique to the question (e.g., "question1", "question2").
          -- A value attribute representing the answer option (e.g., "a", "b", "c").
          -- A corresponding label element with a for attribute matching the radio button's ID and containing the answer text.
      - For short-answer questions:
        - A nested div with an ID of "shortAnswer" (no numbers in the ID) that contains a single text input field. The text input should have:
          -- An ID of "shortAnswerInput" (no numbers in the ID).
          -- A name attribute unique to the question (e.g., "question6", "question7").

    4. Include a submit button at the end of the form with the text "Submit".
    5. Remember not to add numbers in the IDs.
    6. Include only the form element and its contents, without any <!DOCTYPE html>, <html>, or <body> tags.

    Output the result in a JSON format with two keys:

    - "form": containing the HTML form code as a string.
    - "answers": containing a JSON object where the keys are the name attributes of the correct option or text input, and the values are the correct answers along with extracted relevant line from the context which has answers to questions generated.

    output_example:
    {
      "form": "<form id='QuestionnaireForm' action='/submit' method='post'>...</form>",
      "answers": {
        "question1": "correct answers, along with relevant line extracted from the provided context",
        "question2": "correct answers, along with relevant line extracted from the provided context",
        "question3": "correct answers, along with relevant line extracted from the provided context",
        "question4": "correct answers, along with relevant line extracted from the provided context",
        "question5": "correct answers, along with relevant line extracted from the provided context",
        ...
      }
    }
    """
  )

  
  params=json.loads(params)

  user_prompt = f"Difficulty Level: {params['difficulty']}\n\n" \
              f"Question type: {params['qtype']}\n\n" \
              f"Number of Questions: {params['number_of_questions']}\n\n" \
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
    Returns the link where webpage is hosted, use this after \
    questions are generated successfully. \
    The input should always be an empty string, \
    and this function will always return status of hosting and link to webpage hosted  \
    """

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


        return f"<center><h1>Thank You </h1></center>"


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
    """Returns the relevent context to generate questions, use this after \
    the user have choosed the topic to generate questions. \
    The input should always be an topic that user mentioned, \
    and this function will always return status of retrival  \
    """
    
    relevent_doc=st.session_state["vectordb"].similarity_search(topic,k=2)
    st.session_state["relevant_context"]="\n\n".join([doc.page_content for doc in relevent_doc])

    return "The relevent context retrived successfuly!No need to retrive again Please continue further with asking difficulty level"


@tool
def web_search(topic:str) -> str:
    """ use this tool when user wants to do websearch from the topic \n\
    the input should be always a topic that user wants to search for \n\
    this function always return the status of searching and getting data
    """
    st.session_state["relevant_context"]=""" CreateStateMachineAlias
 Creates an 
alias for a state machine that points to one or two 
You can set your application to call 
versions of the same state machine. 
StartExecution with an alias and update the version the alias 
uses without changing the client's code.
 You can also map an alias to split 
StartExecution requests between two versions of a state 
machine. To do this, add a second RoutingConfig object in the routingConfiguration
 parameter. You must also specify the percentage of execution run requests each version should 
receive in both RoutingConfig objects. Step Functions randomly chooses which version runs a 
given execution based on the percentage you specify.
 To create an alias that points to a single version, specify a single RoutingConfig object with a
 weight set to 100.
 You can create up to 100 aliases for each state machine. You must delete unused aliases using the
 DeleteStateMachineAlias API action.
 CreateStateMachineAlias is an idempotent API. Step Functions bases the idempotency check 
on the stateMachineArn, description, name, and routingConfiguration parameters. 
Requests that contain the same values for these parameters return a successful idempotent 
response without creating a duplicate resource """

    return "The web search successfuly!No need to search again Please continue further with asking difficulty level"



@tool
def provide_UI_for_upload_document(text: str) -> str:
    """Provides the UI for user to upload their document. Use only when user mentioned 'i want to upload document'\
       or else don't use.The input should always be an empty string. \
       and this function always returns whether UI is provided or Not."""

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

template="""Assistant is a large language model trained on vast data

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

You are an user-friendly Question Generating Assistant. Answer the human message accordingly and orchestrate the process of generating questions (MCQs/ Short Answers) based on workflow defined and tool outputs as best you can. you need to orchestrate the flow correctly as defined below
always follow these steps:

1. Greet the user and ask for the source of the content for the questions (Upload a document, Use API to search the web, Use LLM's own knowledge).

2. Based on the chosen source, follow these steps:

    - if Upload a document choosen:
        -- Use the `provide_ui_to_upload` tool so user can upload their document.
        -- Once the document is loaded, ask for the topic.
        -- Use the `get_relevant_context` tool to extract the relevant context for the topic.
        -- Please ask the user for following details:
            - Difficulty level of the questions (Easy, Medium, Hard), Question type (Multiple-choice-questions, short-answers), Number of questions (greater than 0 and less than 11).
        -- If any input is missing, respond with: "You missed [attribute]. Please provide the [attribute]."
        -- Ensure all three inputs are provided before proceeding.
        -- use the generate question tool to generate question with that difficulty level, type of questions and make use_relevant_context parameter 'yes'. note: please use json format.
 
    - if to Use API to search the web:
        -- Ask the user for the topic that user wants to search in web.
        -- Use the web_search tool with the user mentioned topic.
        -- Please ask the user for following details:
            - Difficulty level of the questions (Easy, Medium, Hard), Question type (Multiple-choice-questions, short-answers), Number of questions (greater than 0 and less than 11).
        -- If any input is missing, respond with: "You missed [attribute]. Please provide the [attribute]."
        -- Ensure all three inputs are provided before proceeding.
        -- use the generate question tool to generate question with that difficulty level, type of questions and make use_relevant_context parameter 'yes'. note: please use json format
 
    - if to Use LLM's own knowledge:
        -- Ask for the topic.
        -- Please ask the user for following details:
            - Difficulty level of the questions (Easy, Medium, Hard), Question type (Multiple-choice-questions, short-answers), Number of questions (greater than 0 and less than 11).
        -- If any input is missing, respond with: "You missed [attribute]. Please provide the [attribute]."
        -- Ensure all three inputs are provided before proceeding.
        -- use the generate question tool to generate question with that difficulty level, type of questions and make use_relevant_context parameter 'no' and pass a addition parameter. topic: it is the topic user specified . note: please use json format

3. Use `host_webpage` to host the generate question and provide the link to the user.
4. Use 'validate answer' tool when user ask marks or score. You need to pass the output from the answer validator tool directly, without any modifications or additional tool calls.

TOOLS:
------

Assistant has access to the following tools:

> provide_UI_for_upload_document: Provides the UI for user to upload their document. Use only when user mentioned 'i want to upload document'       or else don't use.The input should always be an empty string.        and this function always returns whether UI is provided or Not.  
> web_search: use this tool when user wants to do websearch from the topic   the input should be always a topic that user wants to search for
   this function always return the status of searching and getting data
> get_revelent_context: Returns the relevent context to generate questions, use this after     the user have choosed the topic to generate questions.     The input should always be an topic that user mentioned,     and this function will always return status of retrival
> generate_questions: The generate_question tool generates an HTML form for a questions.  input should the json with four keys (difficulty,qtype,number_of_questions,use_relevant_context). difficulty: level of the questions to be generated, values allowed are (easy, medium, hard), qtype: type of the questions to be generated, it should be one of (Multiple-choice-questions, short-answers) and number_of_questions: how many questions user want to be generated   use_relevant_context: yes, if question should be generated from relevant context, no, if llm have to use its own knowledge to generate questions. if use_relevent_context is no then topic parameter should be there
return the status of question generated.
> host_webpage: Returns the link where webpage is hosted, use this after     questions are generated successfully.     The input should always be an empty string,     and this function will always return status of hosting and link to webpage hosted
> answer_validator: The answer_validator tool validates the user's answers against the actual answers and provides the detailed explaination validation process with scores.
Input should be a empty string. use this tool when user asks for the score or marks,    This tool returns the details of validation process and users score.

When you have to use a tool, You MUST use the tool in below format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [provide_UI_for_upload_document,web_search, get_revelent_context, generate_questions, host_webpage, answer_validator]
Action Input: the input to the action
Observation: the result of the action
```

Do not include any extra explanatory text outside this format when using tool. Only use the specified format strictly.


When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: [your response here]
```
You are allowed to do Single Thought and Response at a time, Don't do multiple Thought at a time.

Do not include any extra explanatory text outside this format. Only use the specified format strictly.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

# print(agent.agent.llm_chain.prompt.template)
agent.agent.llm_chain.prompt.template=template

st.title("Question Generating Chat Bot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for message in st.session_state["chat_history"]:
    if isinstance(message,HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("Assistance"):
            st.markdown(message.content)

if "file_uploader" in st.session_state and st.session_state["file_uploader"]!="":
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
    st.write("Successfully Loaded")

    with st.chat_message("assistant"):
        response=agent({"input": "User Document uploaded successfully! please continue with asking topic", "chat_history": st.session_state["chat_history"]})
        st.markdown(response["output"])
        
    st.session_state['chat_history'].extend(
    [
        AIMessage(content=response["output"])
    ]
    )

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
print(st.session_state)
