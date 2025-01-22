import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun, PubmedQueryRun, ShellTool
from langchain_community.utilities.requests import RequestsWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import SerpAPIWrapper
import os
from dotenv import load_dotenv

load_dotenv()

# st.set_page_config(page_title="Agent ",
#                     page_icon='🤖',
#                     layout='centered',
#                     initial_sidebar_state='collapsed')

# st.header("Web Search 🤖")

# Initialize tools
duckduckgo_search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()
arxiv = ArxivQueryRun()
pubmed = PubmedQueryRun()
requests_tool = RequestsWrapper()
shell_tool = ShellTool()
python_repl = PythonREPLTool()
google=SerpAPIWrapper(search_engine="google", serpapi_api_key=os.getenv("SERP_API_KEY"))
youtube=SerpAPIWrapper(search_engine="youtube", serpapi_api_key=os.getenv("SERP_API_KEY"))



# related to coding or mathematical calculations
tools1=[
    Tool(
        name="python code",
        func=duckduckgo_search.run,
        description="use to run python code or dome some mathematical calculations",
    )
]

#related to real time information
tools2=[
    Tool(
        name="DuckDuckGo",
        func=duckduckgo_search.run,
        description="Search the web for information",
    ),
    Tool(
        name="youtube search",
        func=youtube.run,
        description="use it when you need to search for videos or music over youtube"
    ),
    Tool(
        name="google search",
        func=google.run,
        description="use it when you need to search realtime information or to find links to resources and to search for new and old tech or any related stuff over internet. Also use it to find any links over internet"
    )
]

# use it for research purposes
tools3=[
    Tool(
        name="DuckDuckGo Search",
        func=duckduckgo_search.run,
        description="Search the web for information using DuckDuckGo."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Retrieve detailed information from Wikipedia."
    ),
    Tool(
        name="ArXiv",
        func=arxiv.run,
        description="Search for scientific papers on ArXiv."
    ),
    Tool(
        name="PubMed",
        func=pubmed.run,
        description="Search for biomedical literature on PubMed."
    ),
    Tool(
        name="google search",
        func=google.run,
        description="use it when you need to search realtime information or to find links to resources and to search for new and old tech or any related stuff over internet. Also use it to find any links over internet"
    )
]


llm = ChatGroq(model="llama-3.1-70b-versatile",
            temperature=0.5,
            max_tokens=None,
            timeout=60,
            max_retries=2,
            api_key=os.getenv("GROQ_API_KEY"))

agent1= initialize_agent(llm=llm,
                         tools=tools1,
                         verbose=True,
                         agent="zero-shot-react-description",
                         handle_parsing_errors=True)

agent2= initialize_agent(llm=llm,
                            tools=tools2,
                            verbose=True,
                            agent="zero-shot-react-description",
                            handle_parsing_errors=True)

agent3= initialize_agent(llm=llm,
                            tools=tools3,
                            verbose=True,
                            agent="zero-shot-react-description",
                            handle_parsing_errors=True)

def agent1_run(input_string):
    return agent1.run(input_string)

def agent2_run(input_string):
    return agent2.run(input_string)

def agent3_run(input_string):
    return agent3.run(input_string)

manager_tools=[
    Tool(
        name="agent1",
        func=agent1_run,
        description="if you need to do some matheatical calculations or coding",
    ),
    Tool(
        name="agent2",
        func=agent2_run,
        description="if you need to search real time information",
    ),
    Tool(
        name="agent3",
        func=agent3_run,
        description="if you need to search for research purposes",
    )
    
]


manager= initialize_agent(llm=llm,
                            tools=manager_tools,
                            verbose=True,
                            agent="zero-shot-react-description",
                            handle_parsing_errors=True)

# manager.run("provide a random number (n) and then search for the top n countries with the highest GDP also research on when will india come out of the covid-19 pandemic")

user = st.text_input("Enter your question:")
if st.button("Get Answer"):
    result = manager.run(user)
    st.write("User queury response:")
    st.write(result)
