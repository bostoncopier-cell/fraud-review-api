import os
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
import resend

app = FastAPI()

# Allow calls from your WordPress page
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

# Use env vars set in Render (DO NOT hardcode keys in code)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
resend.api_key = os.environ.get("RESEND_API_KEY")


# ✅ SMOKE TEST ENDPOINT
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/submit")
async def submit(
    transaction_type: str = Form(...),
    contact_email: str = Form(...),
    short_description: str = Form(""),
    files: list[UploadFile] = File(...),
):
    try:
        submission_id = str(uuid.uuid4())

        # Combine some text from uploaded files
        combined_text = ""
        for f in files:
            content = await f.read()
            try:
                combined_text += content.decode("utf-8", errors="ignore")[:5000]
                combined_text += "\n\n"
            except Exception:
                pass

        prompt = f"""
Analyze the following transaction communication for fraud risk.

Transaction Type: {transaction_type}
Description: {short_description}
User Contact Email: {contact_email}

Content:
{combined_text}

Provide:
- Risk Level (Low, Moderate, High)
- Key Findings (bullets)
- Short Assessment
- Recommendation
"""

        # AI analysis
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
            )
            ai_text = response.output_text
        except Exception as e:
            ai_text = f"AI analysis failed: {str(e)}"

        # Email analyst
        try:
            if resend.api_key:
                resend.Emails.send(
                    {
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
                        """,
                    }
                )
        except Exception as e:
            print("Email send error:", e)

        return {
            "ok": True,
            "submission_id": submission_id,
            "message": "Submitted successfully.",
        }

    except Exception as e:
        # This prevents "NetworkError" and shows real error
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )
