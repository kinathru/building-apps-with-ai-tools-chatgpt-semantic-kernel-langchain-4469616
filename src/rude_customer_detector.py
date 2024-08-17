import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
# Challenge: Turning Away Rude Customers
# Build a GPT-4 python app that talks with a user.
# End the conversation if they're being rude

# test case 1 'you're the worst human i've talked to' -> RUDE
# test case 2 'hey how's your day going'
# test case 3 'I like pizza. What do you like?'
# test case 4 'I bite my thumb at you!'
# test case 5 'I think this product doesnt work!' -> RUDE

while True:
    user_input = input("Chat with me : ")

    if user_input == "exit" or user_input == "quit":
        break

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a sentiment classification bot, if the user is rude print just Rude, otherwise print OK"},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=150,
    )

    response_message = response["choices"][0]["message"]
    print(response_message)
    if response_message.content == "Rude":
        print("You are being rude. I'm ending this chat")
        break

    user_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Respond to what user asks"},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=150,
    )
    print(user_response["choices"][0]["message"])
