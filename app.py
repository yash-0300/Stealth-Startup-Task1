import os
import time
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from groq import Groq
from utils import *
from PROMPTS import *
from crewai import Agent, Task, Crew
from crewai_tools import PDFSearchTool, BaseTool
from crewai_tools  import tool
from langchain_community.tools.tavily_search import TavilySearchResults


# API KEYS
os.environ['GROQ_API_KEY'] = 'PASTE YOUR KEY'
os.environ['OPENAI_API_KEY'] = 'PASTE YOUR KEY'
os.environ['TAVILY_API_KEY'] = 'PASTE YOUR KEY'

web_search_tool = TavilySearchResults(k = 3)

@tool
def router_tool(question):
    """Router Function"""
    return 'vectorstore'

def getAgentsTasks(rag_tool, question):

    Router_Agent = Agent(
        role='Router',
        goal='Route user question to a vectorstore or web search',
        backstory=(
        "You are an expert at routing a user question to a vectorstore or web search."
        "Use the vectorstore for questions on concept related to Retrieval-Augmented Generation."
        "You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search."
        ),
        verbose=True,
        allow_delegation=False,
    )

    Retriever_Agent = Agent(
        role="Retriever",
        goal="Use the information retrieved from the vectorstore to answer the question",
        backstory=(
            "You are an assistant for question-answering tasks."
            "Use the information present in the retrieved context to answer the question."
            "You have to provide a clear concise answer."
        ),
        verbose=True,
        allow_delegation=False,
    )

    Grader_agent =  Agent(
        role='Answer Grader',
        goal='Filter out erroneous retrievals',
        backstory=(
        "You are a grader assessing relevance of a retrieved document to a user question."
        "If the document contains keywords related to the user question, grade it as relevant."
        "It does not need to be a stringent test.You have to make sure that the answer is relevant to the question."
        ),
        verbose=True,
        allow_delegation=False,
    )

    hallucination_grader = Agent(
        role="Hallucination Grader",
        goal="Filter out hallucination",
        backstory=(
            "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
            "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
        ),
        verbose=True,
        allow_delegation=False,
    )

    answer_grader = Agent(
        role="Answer Grader",
        goal="Filter out hallucination from the answer.",
        backstory=(
            "You are a grader assessing whether an answer is useful to resolve a question."
            "Make sure you meticulously review the answer and check if it makes sense for the question asked"
            "If the answer is relevant generate a clear and concise response."
            "If the answer gnerated is not relevant then perform a websearch using 'web_search_tool'"
        ),
        verbose=True,
        allow_delegation=False,
    )


    router_task = Task(
        description=("Analyse the keywords in the question {question}"
        "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
        "Return a single word 'vectorstore' if it is eligible for vectorstore search."
        "Return a single word 'websearch' if it is eligible for web search."
        "Do not provide any other premable or explaination."
        ),
        expected_output=("Give a binary choice 'websearch' or 'vectorstore' based on the question"
        "Do not provide any other premable or explaination."),
        agent=Router_Agent,
        tools=[router_tool],
    )

    retriever_task = Task(
        description=("Based on the response from the router task extract information for the question {question} with the help of the respective tool."
        "Use the web_serach_tool to retrieve information from the web"
        "Use the rag_tool to retrieve information from the vectorstore."
        ),
        expected_output=("You should analyse the output of the 'router_task'"
        "You should fetch the answers from the vectorstore or websearch and summarize them in structred way"
        "Return a claer and consise text as response."),
        agent = Retriever_Agent,
        context = [router_task],
    )

    grader_task = Task(
        description=("Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
        ),
        expected_output=("Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
        "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
        "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
        "Do not provide any preamble or explanations except for 'yes' or 'no'."),
        agent=Grader_agent,
        context=[retriever_task],
    )

    hallucination_task = Task(
        description=("Based on the response from the grader task for the quetion {question} evaluate whether the answer is grounded in / supported by a set of facts."),
        expected_output=("Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
        "Respond 'yes' if the answer is in useful and contains fact about the question asked."
        "Respond 'no' if the answer is not useful and does not contains fact about the question asked."
        "Do not provide any preamble or explanations except for 'yes' or 'no'."),
        agent=hallucination_grader,
        context=[grader_task],
    )

    answer_task = Task(
        description=("Based on the response from the hallucination task for the quetion {question} evaluate whether the answer is useful to resolve the question."
        "If the answer is 'yes' return a clear and concise answer."
        "If the answer is 'no' then perform a 'websearch' and return the response"),
        expected_output=("Return a clear and concise response if the response from 'hallucination_task' is 'yes'."
        "Perform a web search using 'web_search_tool' and return ta clear and concise response only if the response from 'hallucination_task' is 'no'."
        "Otherwise respond as 'Sorry! unable to find a valid response'."),
        context=[hallucination_task],
        agent=answer_grader,
    )

    return Router_Agent, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader, router_task, retriever_task, grader_task, hallucination_task, answer_task

st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def main():
    st.set_page_config(page_title="Multi-Person Conversation App", page_icon="🗣️", layout="wide")
    float_init()
    st.title("🗣️ :violet[Multi-Person] Conversation")

    web_search_tool = TavilySearchResults(k = 2)
    # Create a Sidebar to process the PDF
    with st.sidebar:
        st.sidebar.markdown("### Welcome to Multi-Person Conversation AI Assistant!")
        pdf_doc = st.file_uploader("Upload Your PDF Files and Click on the Submit & Process Button", type=['pdf'])
        if pdf_doc:
            temp_path = f"temp_{pdf_doc.name}"
            with open(temp_path, "wb") as f:
                f.write(pdf_doc.getbuffer())
            try:
                rag_tool = PDFSearchTool(pdf=temp_path)
                st.success("PDF processed successfully")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            os.remove(temp_path)
    
        st.sidebar.markdown("### Start Your Conversation!")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Person1 Recording")
            audio_bytes_person1 = audio_recorder(key="person1")
            if audio_bytes_person1:
                # Write the audio bytes to a file
                with st.spinner("Transcribing..."):
                    webm_file_path = "temp_audio.mp3"
                    with open(webm_file_path, "wb") as f:
                        f.write(audio_bytes_person1)
                    
                    transcript = speech_to_text(webm_file_path)
                    if transcript:
                        st.session_state.messages.append({"role": "user", "content": transcript})
                        st.session_state.conversation_history.append({'person1': transcript})
                        os.remove(webm_file_path)

        with col2:
            st.markdown("#### Person2 Recording")
            audio_bytes_person2 = audio_recorder(key="person2")
            if audio_bytes_person2:
                # Write the audio bytes to a file
                with st.spinner("Transcribing..."):
                    webm_file_path = "temp_audio.mp3"
                    with open(webm_file_path, "wb") as f:
                        f.write(audio_bytes_person2)
                    
                    transcript = speech_to_text(webm_file_path)
                    if transcript:
                        st.session_state.messages.append({"role": "assistant", "content": transcript})
                        st.session_state.conversation_history.append({'person2': transcript})
                        os.remove(webm_file_path)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    show_conversation = st.button("Show Conversation History")
    if show_conversation:
        unique_conversations = []
        seen = set()  # To track already added texts

        for i in st.session_state.conversation_history:
            if 'person1' in i:
                text = f"Person1: {i['person1']}"
            elif 'person2' in i:
                text = f"Person2: {i['person2']}"
            else:
                continue
            
            # Add the text to unique_conversations if it hasn't been added already
            if text not in seen:
                unique_conversations.append(text)
                seen.add(text)
        
        # Print the unique conversation texts in order
        conversation_text = ""
        for text in unique_conversations:
            conversation_text += text
            conversation_text += '\n'
            st.write(text)

        # question = getQuestionConversation(conversation_text)
        question = 'What is Machine Learning. Explain in one-two sentence'
        st.write(question)

        Router_Agent, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader, router_task, retriever_task, grader_task, hallucination_task, answer_task = getAgentsTasks(rag_tool, question)
        rag_crew = Crew(
            agents = [Router_Agent, Retriever_Agent, Grader_agent, hallucination_grader, answer_grader],
            tasks = [router_task, retriever_task, grader_task, hallucination_task, answer_task],
            verbose = True,
        )

        inputs = {"question": question}
        result = rag_crew.kickoff(inputs = inputs)
        st.write(result)


if __name__ == "__main__":
    main()
