# Zone Manager Chatbot

A Streamlit-based chatbot for querying vehicle and dealer data using natural language.

## Features
- Natural language SQL query generation
- Ford LLM integration
- Langfuse tracking
- Multi-dealer data analysis

## Deployment

### Prerequisites
- Google Cloud SDK installed
- Ford LLM credentials
- GCP project with required APIs enabled

### Deploy to Google Cloud Run

1. Copy `deploy.template.sh` to `deploy.sh`
2. Update `deploy.sh` with your actual credentials
3. Run: `./deploy.sh`

### Local Development

1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `streamlit run main.py`

## Environment Variables

Required environment variables:
- `FORDLLM_CLIENT_ID`
- `FORDLLM_CLIENT_SECRET`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_HOST`
