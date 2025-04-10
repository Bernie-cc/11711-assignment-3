import json
import os
import openai

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key="",  # Replace with your actual API key
    base_url="https://cmu.litellm.ai",
)

def request_api(user, prompt):
    """
    Sends a request to the API with the given prompt and returns the response content.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Replace with the appropriate model name
        messages=prompt
    )
    response_content = response.choices[0].message.content
    return user, response_content

def main():
    """
    Main function to loop through user prompts and save API responses.
    """
    # Load user prompts from reviewer_prompts.json
    with open("reviewer_prompts.json", "r") as f:
        user_prompts = json.load(f)

    results = {}
    for user, prompt in user_prompts.items():
        user, response_content = request_api(user, prompt)
        results[user] = response_content

    # Save all responses into a JSON file
    output_file = "rank_results.json"
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    main()
