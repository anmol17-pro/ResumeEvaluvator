import os
from pathlib import Path 
from dotenv import load_dotenv 
from groq import Groq 

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API KEY KAHA HAI BETE")

client = Groq(api_key = my_api_key)

model = "llama-3.3-70b-versatile"

# Structered it in json format 

from pydantic import BaseModel
class Ticket(BaseModel):
    name: str
    email: str
    issue: str
    address: str 

schema = Ticket.model_json_schema()

response_format = {
    "type": "json_object"
}

system_pompt = f"""
Extract the personal information from the ticket strictly based on this schema an give me a json output
{schema}
"""
 
text = "Hello My self Anmol. I have an iphone which is not working at all. My address is Haryana. My email is thakralamol17@gmail.com My contact number is 95043XXXX please take my complaint and kindly solve my problem" 

prompt = f"""
This is the customer tickes. Please extract the personal information from this.{text}
"""

messages= [
    {
    "role": "system",
    "content": system_pompt
},
    {"role": "system",
    "content": prompt
    }
]

response = client.chat.completions.create(model = model, messages=messages, response_format=response_format)
answer = response.choices[0].message.content


# isko padna kaise hai 
import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file) 

print(ticket.name)
print(ticket.email)
print(ticket.issue)

