from mail_functions import retrieve_mails, send_email
from config import *

import re
from datetime import datetime, timedelta
from tqdm import tqdm

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

def daily_update(user: str, send_mail: bool = True):

    texts = retrieve_mails(user,datetime.now()+timedelta(days=-1),with_headers=False)

    llm = Ollama(base_url=OLLAMA_BASE_URL,model='llama3.1')

    summary_template = """You are an AI language model assistant. Your task is to summarize the following document into 2 to 3 lines. If the document is empty just output an empty string.
                        You will frame the sentences in passive voice.  You will not provide any introductory line.
                        {document}"""

    summaries = []

    for text in tqdm(texts):
        cleaned_text = re.sub(r'\\.', '', text)

        summary_prompt = ChatPromptTemplate.from_template(summary_template)

        summary_chain = summary_prompt | llm
        
        summary = summary_chain.invoke({'document':cleaned_text})

        summaries.append(summary)

    summaries = list(map(lambda x: '•' + x,summaries))

    summaries = '\n\n'.join(summaries)

    if send_mail:
        send_email(from_= USER, to_= USER, body= summaries,subject= "Daily Email Summaries")

if __name__ == '__main__':
    daily_update(USER)