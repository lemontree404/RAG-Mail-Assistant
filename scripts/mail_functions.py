import os
import base64
from datetime import datetime, timedelta
import email.utils as eut
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import Union

import asyncio
import aiohttp

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from email.mime.text import MIMEText

from config import *

def parse_payload(payload):
    """
    Decodes and extracts the text content from an email payload.

    Args:
        payload (dict): The email payload containing the body data and MIME type.

    Returns:
        str: The decoded and extracted text content of the email. Returns an empty string if decoding fails.
    """
    try:
        text = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')

        if payload['mimeType'] == 'text/html' or bool(BeautifulSoup(text, "html.parser").find()):
            soup = BeautifulSoup(text, 'html.parser')
            body = soup.get_text()
            return body.replace('\n\n','')
        elif payload['mimeType'] == 'text/plain':
            body = text
            return body
    
    except:
        return ""

def get_header(headers: dict, name: str):
    """
    Retrieves the value of a specific header from an email's headers.

    Args:
        headers (dict): A dictionary containing the email headers.
        name (str): The name of the header to retrieve.

    Returns:
        str or None: The value of the specified header, or None if not found.
    """
    for header in headers:
        if header['name'].lower() == name.lower():
            return header['value']
    return None

def compile_email(service, user_id: str, message_id: str, with_headers: bool = True) -> str:
    """
    Retrieves and compiles the full content of an email, including headers and body.

    Args:
        service: The Gmail API service instance used to access the user's emails.
        user_id (str): The user's Gmail ID.
        message_id (str): The ID of the specific email to retrieve.

    Returns:
        Tuple[datetime, str]: A tuple containing the email's date as a `datetime` object 
        and the compiled email content as a string.
    """    
    mail = service.users().messages().get(userId=user_id, id=message_id, format='full').execute()

    # Extract headers
    headers = mail['payload']['headers']
    received = get_header(headers, 'Received')
    subject = get_header(headers, 'Subject')
    from_email = get_header(headers, 'From')
    to_email = get_header(headers, 'To')
    date = get_header(headers, 'Date')
    message_id = get_header(headers, 'Message-ID')

    # Extract body (assuming plain text or HTML)
    body = ''
    if 'parts' in mail['payload']:
        for part in mail['payload']['parts']:

            body = parse_payload(part)
    else:
        body = parse_payload(mail['payload'])

    # Compile everything into a single string
    if with_headers:
        email_string = f"""\
        Received: {received}
        Subject: {subject}
        From: {from_email}
        To: {to_email}
        Date: {date}
        Message-ID: {message_id}

        {body}
        """
    
    else:
        email_string = body

    date_tuple = eut.parsedate_tz(date)
    date = datetime.fromtimestamp(eut.mktime_tz(date_tuple))

    # Print the compiled string
    return (date,email_string)

def get_creds():
    """
    Obtains and returns the user's credentials for accessing the Gmail API.

    This function checks if valid credentials exist in a local file ("token.json").
    If not, it initiates the OAuth2 flow to obtain new credentials and saves them
    for future use.

    Returns:
        Credentials: The authenticated credentials for the Gmail API.
    """
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly','https://www.googleapis.com/auth/gmail.send']
    
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "creds.json", SCOPES
        )
        creds = flow.run_local_server(port=0)
    
    with open("token.json", "w") as token:
        token.write(creds.to_json())
    
    return creds

def retrieve_mails(user: str, date_lim: datetime, batch_size: int = 50, messages_per_request: int = 500, with_headers: bool = True):
    """
    Retrieves emails from the Gmail API for a specified user, filtering by date and processing in batches.

    Args:
        user (str): The user's Gmail ID.
        date_lim (datetime): The date limit; only emails received on or after this date are retrieved.
        batch_size (int, optional): The number of message IDs to process per batch. Defaults to 50.
        messages_per_request (int, optional): The maximum number of messages to retrieve per API request. Defaults to 500.

    Returns:
        List[str]: A list of compiled email contents as strings, sorted by date in descending order.
    """

    creds = get_creds()
        
    service = build('gmail', 'v1', credentials=creds)

    async def process_messages(message_ids):
        """
        Processes a list of message IDs, fetching and compiling email content.

        Args:
            message_ids (List[str]): A list of email message IDs to process.

        Returns:
            Tuple[List[Tuple[datetime, str]], bool]: A tuple where the first element is a list of tuples containing
            the email date and compiled content, and the second element is a boolean indicating whether any email
            was received before the specified date limit (True if processing should stop).
        """
        compiled_emails = []
        for message_id in message_ids:
            date, mail = compile_email(service,user, message_id, with_headers)
            if date < date_lim:
                return compiled_emails, True  # Stop if date is earlier than the limit
            compiled_emails.append((date, mail))
        # print("Batch Complete")
        return compiled_emails, False

    async def async_driver(response):
        """
        Handles asynchronous processing of message IDs in batches and compiles email data.

        Args:
            response (dict): The response from the Gmail API containing a list of message IDs.

        Returns:
            List[Tuple[List[Tuple[datetime, str]], bool]]: A list of tuples, where each tuple contains
            a list of compiled emails and a boolean indicating whether processing should stop.
        """
        messages = response.get('messages', [])
        message_ids = [msg['id'] for msg in messages]

        # Process message IDs in batches
        async with aiohttp.ClientSession() as session:
            # Create batches of message IDs
            batches = [message_ids[i:i + batch_size] for i in range(0, len(message_ids), batch_size)]
            tasks = [process_messages(batch) for batch in batches]

            resp = await asyncio.gather(*tasks)
            return resp
    
    all_emails = []
    break_flag = False

    response = service.users().messages().list(userId=user, maxResults=messages_per_request).execute()
    print(f'{messages_per_request} messages fetched')

    while not break_flag:
        async_responses = asyncio.run(async_driver(response))
        temp = [ mail for resp in async_responses for mail in resp[0]]
        temp.sort(key = lambda x: x[0], reverse=True)
        all_emails += [i[1] for i in temp]

        flags = [resp[1] for resp in async_responses]
        print(flags)
        break_flag = any(flags)

        if 'nextPageToken' in response and not break_flag:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user, maxResults=500, pageToken=page_token).execute()
            print(f'{messages_per_request} messages fetched')
        else:
            break
        
    print("Mails Retrieved")
    return all_emails

def send_email(from_: str, to_: str, body: str, subject: str = None):
    
    creds = get_creds()
    service = build('gmail', 'v1', credentials=creds)

    # Create the email content
    message = MIMEText(body)
    message['to'] = to_
    message['from'] = from_
    if subject is None:
        message['subject'] = ''
    else:
        message['subject'] = subject

    # Encode the email message
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    # Create the message body
    body = {
        'raw': raw
    }

    # Send the email
    try:
        sent_message = service.users().messages().send(userId=from_, body=body).execute()
        print(f"Email sent to {to_}. Message ID: {sent_message['id']}")
    except Exception as error:
        print(f"An error occurred while sending email: {error}")

def main():
    emails = retrieve_mails(user=USER, date_lim=datetime.now() + timedelta(days=-1))
    emails = '\n'.join(emails)
    with open('mails.txt','w') as file:
        file.write(emails)

if __name__ == "__main__":
    main()  