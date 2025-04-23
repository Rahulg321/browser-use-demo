from langchain_openai import ChatOpenAI
from browser_use import Agent,  Browser, BrowserConfig, Controller
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
load_dotenv()

import asyncio


class TeamMember(BaseModel):
    name: str
    designation: str


class TeamMembers(BaseModel):
    team_members: List[TeamMember]


controller = Controller(output_model=TeamMembers)

browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        browser_binary_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    )
)

initial_actions = [
	{'open_tab': {'url': 'https://www.darkalphacapital.com/'}},
]

# llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', temperature=0, timeout=100)
llm = ChatOpenAI(model='gpt-4o', temperature=0, timeout=100)

async def main():
    try:
        agent = Agent(
            task="Go to https://www.darkalphacapital.com/ and scrape all the team members from the page specifically the name, title, and image url of the team members",
            llm=llm,
            browser=browser,
            controller=controller,
            initial_actions=initial_actions,
        )
        result = await agent.run()
        print(result.final_result())
        data = result.final_result()
        parsed:TeamMembers = TeamMembers.model_validate_json(data)
        print(parsed.team_members)
    except Exception as e:
        print(f"Error occurred during execution: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        try:
            await browser.close()
        except Exception as e:
            print(f"Error closing browser: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())