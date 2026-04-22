#!/bin/bash

# Ensure script stops on first error
set -e

echo "Building Docker image locally..."
docker build -t building-codes .

echo "To deploy to Google Cloud Run, please run the following command:"
echo "gcloud run deploy building-codes \\"
echo "  --source . \\"
echo "  --region us-central1 \\"
echo "  --allow-unauthenticated \\"
echo "  --set-env-vars RAG_WORKING_DIR=/app/lightrag_storage"
echo ""
echo "Note: Make sure your .env file is configured, but do not commit it. You may need to pass --set-secrets for OPENAI_API_KEY in production."
