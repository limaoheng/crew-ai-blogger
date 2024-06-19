import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool  # SerperDevTool for search
import uvicorn
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Define the FastAPI app
app = FastAPI()


# Define the input model for the request
class TopicRequest(BaseModel):
    topic: str


# Create the Serper search tool
search_tool = SerperDevTool()

LLM = ChatGroq(model="llama3-70b-8192")

# Define the Senior Researcher agent
researcher = Agent(
    role='Senior Researcher',
    goal='Uncover groundbreaking technologies in {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change"
        "the world."
    ),
    tools=[search_tool],
    max_iter=5,
    llm=LLM
)

# Define the Writer agent
writer = Agent(
    role='Writer',
    goal='Narrate compelling tech stories about {topic}',
    verbose=True,
    memory=True,
    llm=LLM,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    allow_delegation=False
)

# Define the Research Task
research_task = Task(
    description=(
        "Identify the next big trend in {topic}."
        "Focus on identifying pros and cons and the overall narrative."
        "Your final report should clearly articulate the key points,"
        "its market opportunities, and potential risks."
    ),
    expected_output='A comprehensive 3-paragraph report on the latest trends in {topic}.',
    tools=[search_tool],
    agent=researcher
)

# Define the Writing Task
write_task = Task(
    description=(
        "Compose an insightful article on {topic}."
        "Focus on the latest trends and how they're impacting the industry."
        "This article should be easy to understand, engaging, and positive."
    ),
    expected_output='A 4-paragraph article on {topic} advancements formatted as markdown.',
    agent=writer,
    async_execution=False,
    output_file='new-blog-post-about-{topic}.md'
)


@app.post("/generate-blog")
async def generate_blog(request: TopicRequest):
    # Assemble the crew with the defined agents and tasks
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential  # Sequential task execution
    )

    # Run the crew with the given topic
    try:
        result = crew.kickoff(inputs={'topic': request.topic})
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)