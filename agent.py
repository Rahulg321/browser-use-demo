# Import necessary libraries
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, Controller
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError # Added ValidationError for better error handling
from typing import List, Optional # Optional might be useful if some fields aren't always present
import pandas as pd
import asyncio
import os # To securely get environment variables
import traceback # For detailed error logging
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables from .env file
load_dotenv()

# --- Pydantic Models for Data Structure ---

class TeamMember(BaseModel):
    """Represents a team member."""
    name: str
    designation: str

class TeamMembers(BaseModel):
    """Represents a list of team members."""
    team_members: List[TeamMember]

class Listing(BaseModel):
    """Represents a single deal listing."""
    title: str
    state: str
    revenue: Optional[float] = None
    ebitda: Optional[float] = None 
    asking_price: Optional[float] = None

class Listings(BaseModel):
    """Represents a list of deal listings."""
    listings: List[Listing]


async def main():
    """
    Main asynchronous function to run the web scraping agent.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GEMINI_API_KEY")
    
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    if not google_api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return


    controller = Controller(output_model=Listings)

   
    chrome_path = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
    if not os.path.exists(chrome_path):
        chrome_path_alt = 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
        if os.path.exists(chrome_path_alt):
            chrome_path = chrome_path_alt
            print(f"Using Chrome path: {chrome_path}")
        else:
            print(f"Warning: Chrome executable not found at primary path ({chrome_path}) or alternative path ({chrome_path_alt}).")
            print("Attempting to run without specifying browser path (might work if Chrome is in system PATH)...")
            chrome_path = None # Let playwright try to find it
    else:
        print(f"Using Chrome path: {chrome_path}")

    browser = Browser(
        config=BrowserConfig(
            browser_binary_path=chrome_path,
            # Consider increasing timeouts if the target site is slow
            # page_load_timeout=90000, # milliseconds (e.g., 90 seconds)
            # action_timeout=45000 # milliseconds (e.g., 45 seconds)
        )
    )

    initial_url = 'https://www.edisonba.com/pages/listings'
    initial_actions = [
        {'open_tab': {'url': initial_url}},
    ]

    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', temperature=0, timeout=180, api_key=google_api_key)


    task_description = (
        f"Your goal is to scrape all business listings from {initial_url} and its subsequent pages. "
        "1. On each page, scrape every visible deal listing, extracting its 'title', 'state', 'revenue', 'ebitda', and 'asking_price' (treat missing values as N/A). "
        "2. Look for a pagination control labeled 'Next', '>', or the next page number; verify it exists and is clickable (not disabled or representing the current page). "
        "3. If a clickable 'Next' link/button is found, click it and repeat from step 1. "
        "4. If no clickable 'Next' element exists, stop immediately and return all listings. "
        "5. Finally, compile *all* listings into a single JSON object under the key 'listings', matching the required schema."
    )

    # --- Run the Agent ---
    scraped_listings = []
    try:
        print("Initializing scraping agent...")
        agent = Agent(
            task=task_description,
            llm=llm,
            browser=browser,
            controller=controller,
            initial_actions=initial_actions,
            validate_output=True,
            override_system_message=(
                "You are an autonomous web agent. You must stop execution if:\n"
                "1. There are no more clickable pagination elements (e.g., 'Next' is missing, disabled, or grayed out).\n"
                "2. You reach the end of the listings or the page doesn't change after a click.\n"
                "Scrape only visible data. Do not go to unrelated pages. Be efficient and deterministic."
                )
        )

        print(f"Running agent to scrape listings starting from: {initial_url}")
        result = await agent.run(max_steps=5)
        print("Agent finished running.")

        # --- Process Results ---
        final_data_json = result.final_result()

        if final_data_json:
            print("Parsing scraped data...")
            try:
                # Validate the JSON data against the Pydantic model
                parsed: Listings = Listings.model_validate_json(final_data_json)
                print("--------------------------------")
                print("Parsed Listings:")
                print("--------------------------------")

                for listing in parsed.listings:
                    # Append the validated listing object directly
                    scraped_listings.append(listing)
                    # Print details (including EBITDA)
                    print(f"  Title:          {listing.title}")
                    print(f"  State:          {listing.state}")
                    print(f"  Revenue:        {listing.revenue if listing.revenue is not None else 'N/A'}")
                    print(f"  EBITDA:         {listing.ebitda if listing.ebitda is not None else 'N/A'}") # Added EBITDA print
                    print(f"  Asking Price:   {listing.asking_price if listing.asking_price is not None else 'N/A'}")
                    print("  ---")

            except ValidationError as ve:
                print(f"\nError: Failed to validate the scraped data against the Listings model.")
                print(f"Validation Errors: {ve}")
                print("\nRaw JSON data received from agent:")
                print(final_data_json) # Print raw data for debugging
            except Exception as e:
                 print(f"\nError processing agent results: {e}")
                 print("\nRaw JSON data received from agent:")
                 print(final_data_json) # Print raw data for debugging

        else:
            print("Agent did not return any final data.")

        print("--------------------------------")
        print(f"Total listings scraped: {len(scraped_listings)}")
        print("--------------------------------")

        # --- Save to Excel ---
        if scraped_listings:
            print("Saving listings to Excel...")
            try:
                # Use model_dump() to convert Pydantic models to dictionaries for DataFrame
                df = pd.DataFrame([listing.model_dump() for listing in scraped_listings])
                excel_filename = "edisonba_listings.xlsx" # Changed filename
                df.to_excel(excel_filename, index=False)
                print(f"Listings successfully saved to {excel_filename}")
            except Exception as e:
                print(f"Error saving listings to Excel: {e}")
        else:
            print("No listings were scraped, skipping Excel save.")

        print("--------------------------------")

    except Exception as e:
        # Catch potential errors during agent execution or setup
        print(f"\n--- An unexpected error occurred during agent execution ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("----------------------------------------------------------")

    finally:
        # --- Cleanup ---
        print("Closing browser...")
        try:
            await browser.close()
            print("Browser closed successfully.")
        except Exception as e:
            print(f"Error closing browser: {str(e)}")

# --- Script Entry Point ---
if __name__ == "__main__":
    print("Starting script execution...")
    asyncio.run(main())
    print("Script execution finished.")
