import os
import sys
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

# Capture UserInput research_topic and research_question for the task
research_topic = sys.argv[1]
research_question = sys.argv[2]

print(f"Received research topic: {research_topic}")
print(f"Received research question: {research_question}")

# Set your API keys as environment variables
os.environ["SERPER_API_KEY"] = "" # serper.dev API key
os.environ["OPENAI_API_KEY"] = "NA"

# Importing ChatOpenAI
llm = ChatOpenAI(
    model = "llama3",
    base_url = "http://localhost:11434/v1")

# Loading Tools
search_tool = SerperDevTool()

# Define your agents with roles, goals, tools, and additional attributes
researcher = Agent(
  role='Senior Research Analyst',
  goal=f'Uncover cutting-edge developments in {research_topic}',
  backstory=(
    "You are a Senior Research Analyst at a leading tech think tank."
    f"Your require to investigate the {research_question} until you satisfy the Tech Content Writer"
    "You MUST use the search tool in order to collect sufficent data and present your technical insights to he Tech Content Writer."
  ),
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=llm,
  max_rpm=100
)
writer = Agent(
  role='Tech Content Writer',
  goal='Craft a compelling content on the Senrio Research Analyst findings',
  backstory=(
    "You are a renowned Tech Content Writer, known for your sharp and engaging articles on technology and innovation."
    "With a deep understanding of the technical writing, you transform complex concepts into compelling narratives."
  ),
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=llm,
  cache=True, # Disable cache for this agent
)

# Create tasks for your agents
task1 = Task(
  description=(
    f"Conduct a comprehensive analysis of {research_topic}, focusing on {research_question}"
    "Identify key trends, breakthrough technologies, and highlight potential industry disruption."
    "Compile your findings in a detailed report."
  ),
  expected_output=f'A comprehensive full report on {research_topic} focusing ont the {research_question}',
  agent=researcher,
  human_input=False,
)

task2 = Task(
  description=(
    "Using the insights from the researcher's report, develop an engaging blog post that highlights the most significant findings."
    "Your post should be informative yet accessible, catering to a tech-savvy audience."
    "Aim for a narrative that captures the essence of these breakthroughs and their implications for the future."
    "As a final step you are to provide a tabular formatted markdown file with the content of the blog post including theme, positive and negative implications."
  ),
  expected_output=f'Provide markdown formatted blog post responding to the {research_question}, if tables are required make sure tabular markdown is used',
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=1
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)


