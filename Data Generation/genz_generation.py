import sys
import subprocess
import os


    # Try importing again
from groq import Groq
# from dotenv import load_dotenv


# Initialize Groq Client
# load_dotenv('proj.env')
# groq_api_key = os.getenv('GROQ_API')

def Genz_generation(text, groq_api_key,model):
    client = Groq(api_key=groq_api_key)
    prompt = f"""
    You are a Gen Z slang expert. Your sole task is to translate the following text into Gen Z slang **without changing its meaning, tone, or context**, and **without altering any nouns or pronouns**.

    Follow these STRICT rules:

    1. Do NOT include any explanations, labels, intros, or summaries.
    2. ONLY translate the text into Gen Z slang — keep the structure and original message intact.
    3. Do NOT prefix or suffix the output with anything — your response must be JUST the translated text.
    4. NO emojis, hashtags, or any non-textual symbols.
    5. Use **varied** Gen Z slang — don't overuse a single term or rely heavily on one phrase.
    6. Keep the meaning, tone, and intent of the original text 100% intact.
    7. Do NOT use the words "lowkey" or "highkey" in the translation.

    Incorporate Gen Z slang naturally and diversify your usage. Avoid repeating the same slang word in the same response.

    Example Gen Z slang to use (as appropriate):
    Bet, Bussin’, Cap, No cap, Drip, Slay, It’s giving, Periodt, Stan, W, L, Ratio, Rizz, Mid, Sus, Dead, Sheesh, Main character, Ghosting, Ick, Vibe check, Yeet, Glow up, Cancelled, Snack, Simp, Flex, Hits different, Finsta, Iykyk.

    Now translate this text:
    {text}
    """




    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert genz word user."},
                {"role": "user", "content": prompt}
            ],
            model=model
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return f"Failed to generate: {str(e)}"


# summarized_text = summarize_text("Hey, I was just thinking about our trip last summer—remember how we got lost on the hike and ended up finding that amazing little waterfall? Honestly, that whole week was one of the best times I’ve had in a long while. We should definitely plan something like that again soon, maybe somewhere new this time.")
# print(summarized_text)