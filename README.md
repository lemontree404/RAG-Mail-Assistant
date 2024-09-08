# RAG Chat and Daily Email Summaries

This repository contains Python scripts to enable two main functionalities:
1. **RAG Chat (`chat.py`)**: Engage in a Retrieval-Augmented Generation (RAG) chat.
2. **Daily Email Summaries (`daily_updates.py`)**: Can be used to set up a cron job that sends you daily email summaries.

## Directory Structure

- `scripts/` - Contains the main Python scripts:
  - `chat.py` - Script to initiate a RAG-based chat.
  - `daily_updates.py` - Script to configure a cron job for sending daily email summaries.

## Prerequisites

Make sure you have the following installed before running the scripts:

- [Ollama](https://ollama.com/) (for running local LLM models)
- PostgreSQL (psql)

## Setup and Usage

### 1. RAG Chat

To start the RAG chat, run the following command from the `scripts` directory:

```bash
python3 chat.py
```
### 2. Daily Updates

To set up a cronjob that sends you daily email summaries to your email, run the following command from the `scripts` directory:

```bash
crontab -e
* * * * * python ../path_to_repo/scripts/daily_updates.py
```
