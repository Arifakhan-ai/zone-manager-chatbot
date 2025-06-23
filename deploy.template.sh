

# =============================================================================
# REPLACE ALL VALUES BELOW WITH YOUR ACTUAL CREDENTIALS
# =============================================================================

# GCP Project Settings
PROJECT_ID="your-gcp-project-id"

# Ford LLM Credentials
FORDLLM_CLIENT_ID="your-fordllm-client-id"
FORDLLM_CLIENT_SECRET="your-fordllm-client-secret"

# Langfuse Credentials  
LANGFUSE_SECRET_KEY="your-langfuse-secret-key"
LANGFUSE_PUBLIC_KEY="your-langfuse-public-key"

# =============================================================================
# DEPLOYMENT SCRIPT (DO NOT MODIFY BELOW THIS LINE)
# =============================================================================

echo "Setting project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

echo "Creating secrets in Google Secret Manager..."

# Create secrets (only run these once)
echo "$FORDLLM_CLIENT_SECRET" | gcloud secrets create fordllm-client-secret --data-file=- --project=$PROJECT_ID 2>/dev/null || echo "Secret fordllm-client-secret already exists"

echo "$LANGFUSE_SECRET_KEY" | gcloud secrets create langfuse-secret-key --data-file=- --project=$PROJECT_ID 2>/dev/null || echo "Secret langfuse-secret-key already exists"

echo "$LANGFUSE_PUBLIC_KEY" | gcloud secrets create langfuse-public-key --data-file=- --project=$PROJECT_ID 2>/dev/null || echo "Secret langfuse-public-key already exists"

echo "Enabling required APIs..."
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable secretmanager.googleapis.com --project=$PROJECT_ID

echo "Deploying to Cloud Run..."

gcloud run deploy zone-manager-chatbot \
    --source . \
    --project $PROJECT_ID \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 10 \
    --update-secrets="FORDLLM_CLIENT_SECRET=fordllm-client-secret:latest,LANGFUSE_SECRET_KEY=langfuse-secret-key:latest,LANGFUSE_PUBLIC_KEY=langfuse-public-key:latest" \
    --set-env-vars="FORDLLM_CLIENT_ID=$FORDLLM_CLIENT_ID,LLM_TOKEN_ENDPOINT=https://login.microsoftonline.com/c990bb7a-51f4-439b-bd36-9c07fb1041c0/oauth2/v2.0/token,LLM_SCOPE=api://6af47983-2540-43ae-89ff-4b93bf4eeb33/.default.,API_HOST=api.pd01i.gcp.ford.com,PROXY_ENDPOINT=http://internet.ford.com:83,LLM_API_ENDPOINT=https://api.pivpn.core.ford.com/fordllmapi/api/v1/chat/completions,MODEL=gpt-4o-mini-2024-07-18,LANGFUSE_HOST=https://us.cloud.langfuse.com"

echo "Deployment complete!"
echo "Your app should be available at the URL shown above."
