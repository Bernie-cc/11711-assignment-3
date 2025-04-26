import json
import os
import openai
import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank", "-n",
        type=int,
        default=1,
        help="Number of reranking iterations (>=1)."
    )
    args = parser.parse_args()
    rerank_times = args.rerank

    # Load the *initial* user prompts
    with open("reviewer_prompts.json", "r", encoding="utf-8") as f:
        user_prompts = json.load(f)

    all_results = {}

    for user_id, base_prompt in user_prompts.items():
        # We’ll clone the base prompt for each user
        history = list(base_prompt)
        user_iters = []

        for it in range(rerank_times):
            # 1) Call the API
            response = request_api(history)
            user_iters.append(response)

            # 2) If more rounds remain, append assistant + re-ranking instruction
            if it < rerank_times - 1:
                history.append({"role": "assistant", "content": response})
                history.append({
                    "role": "user",
                    "content": (
                        "Please refine your ranking based on the previous results. "
                        "Keep the same JSON format and only return the updated rank string."
                    )
                })

        # Save the list of N responses for this user
        all_results[user_id] = user_iters

    # Dump everything to disk
    with open("rank_results.json", "w", encoding="utf-8") as out_f:
        json.dump(all_results, out_f, indent=2, ensure_ascii=False)

    print(f"Done. Ran {rerank_times} iterations for {len(user_prompts)} users.")

if __name__ == "__main__":
    main()
