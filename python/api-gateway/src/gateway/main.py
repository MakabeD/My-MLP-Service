from fastapi import FastAPI, HTTPException, Depends, Header
import httpx
import google.auth.transport.requests
import google.oauth2.id_token
import os
from fastapi.middleware.cors import CORSMiddleware
from churn_schemas import ChurnInput
from credit_schemas import CreditScoringInput

API_KEY=os.getenv("SECRET_API_KEY", "1")

def verify_api_key(x_api_key: str = Header(None, description="API KEY")):

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Access denied: Invalid or missing API key"
        )
    return x_api_key

app = FastAPI(
    title="API Gateway",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Cloud Run service URL
CHURN_URL = "https://api-mobile-churn-1025838794814.us-east1.run.app"
CREDIT_URL="https://api-credit-scoring-1025838794814.us-east1.run.app"

def get_google_token(audience_url: str) -> str | None:
    """
    Retrieve a Google ID token to authenticate
    requests between Cloud Run services.
    """
    try:
        auth_request = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(auth_request, audience_url)
        return token
    except Exception as error:
        print(f"Token retrieval failed: {error}")
        return None


@app.post("/api/mlp/predict/churn", dependencies=[Depends(verify_api_key)])
async def route_to_churn(payload: ChurnInput):
    """
    Forward prediction requests to the private churn model service.
    """

    token = get_google_token(CHURN_URL)

    if not token:
        raise HTTPException(
            status_code=500,
            detail="Gateway could not obtain Google credentials."
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{CHURN_URL}/predict",
            json=payload.model_dump(mode='json', by_alias=True),
            headers=headers
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Private model request failed: {response.text}"
        )

    return response.json()

@app.post("/api/mlp/credit/scoring",dependencies=[Depends(verify_api_key)])
async def route_to_credit(payload: CreditScoringInput):
    """
    Forward prediction requests to the private credit scoring  model service.
    """

    token = get_google_token(CREDIT_URL)

    if not token:
        raise HTTPException(
            status_code=500,
            detail="Gateway could not obtain Google credentials."
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{CREDIT_URL}/predict",
            json=payload.model_dump(mode='json', by_alias=True),
            headers=headers
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Private model request failed: {response.text}"
        )

    return response.json()