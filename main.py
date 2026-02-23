import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import resend

app = FastAPI()

# 👇 CHANGE THIS TO YOUR WORDPRESS DOMAIN
ALLOWED_ORIGINS = [
    "https://sales101.org",
    "https://www.sales101.org",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANALYST_EMAIL = "bostoncopier@gmail.com"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

resend.api_key = os.environ.get("RESEND_API_KEY")

@app.post("/api/submit")
async def submit(
    transaction_type: str = Form(...),
    contact_email: str = Form(...),
    short_description: str = Form(""),
    files: list[UploadFile] = File(...)
):
    submission_id = str(uuid.uuid4())

    combined_text = ""

    for f in files:
        content = await f.read()
        try:
            combined_text += content.decode("utf-8", errors="ignore")[:5000]
        except:
            pass

    # --- AI FRAUD ANALYSIS ---
    prompt = f"""
    Analyze the following transaction communication for fraud risk.

    Transaction Type: {transaction_type}
    Description: {short_description}

    Content:
    {combined_text}

    Provide:
    - Risk Level (Low, Moderate, High)
    - Key Findings (bullets)
    - Short Assessment
    - Recommendation
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    ai_text = response.output_text

    # --- EMAIL TO BOSTONCOPIER ---
if resend.api_key:
    resend.Emails.send({
        "from": "Fraud Review <onboarding@resend.dev>",
        "to": [ANALYST_EMAIL],
        "subject": f"Fraud Review Submission {submission_id}",
        "html": f"""
        <h2>Fraud Review Submission</h2>
        <p><b>Submission ID:</b> {submission_id}</p>
        <p><b>Transaction Type:</b> {transaction_type}</p>
        <p><b>User Contact Email:</b> {contact_email}</p>
        <p><b>Description:</b> {short_description}</p>
        <hr/>
        <pre>{ai_text}</pre>
        """
    })

    return {
        "ok": True,
        "submission_id": submission_id,
        "message": "Submitted successfully."
    }
