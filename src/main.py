from fastapi import FastAPI
from src.api.v1 import news as news_router

app = FastAPI(
    title="CryptoPulse AI API",
    description="API for fetching data and running models for the CryptoPulse AI app.",
    version="1.0.0"
)

app.include_router(news_router.router, prefix="/v1/news", tags=["News"])

@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"message": "Welcome to the CryptoPulse AI API!"}