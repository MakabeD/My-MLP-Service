from fastapi import FastAPI, HTTPException
import httpx
import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI(
    title="API Gateway",
    version="1.0"
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


@app.post("/api/mlp/predict/churn")
async def route_to_churn(payload: dict):
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

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{CHURN_URL}/predict",
            json=payload,
            headers=headers
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Private model request failed: {response.text}"
        )

    return response.json()