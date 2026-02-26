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

# Environment variables (set these in Render)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
resend.api_key = RESEND_API_KEY


# ✅ SMOKE TEST ENDPOINT
@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(OPENAI_API_KEY),
        "resend_configured": bool(RESEND_API_KEY),
    }


@app.post("/api/submit")
async def submit(
    transaction_type: str = Form(...),
    contact_email: str = Form(...),
    short_description: str = Form(""),
    files: list[UploadFile] = File(...),
):
    try:
        submission_id = str(uuid.uuid4())

        # Combine some text from uploaded files (best-effort)
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

        # -------------------------
        # AI analysis (optional)
        # -------------------------
        ai_text = ""
        ai_error = None

        if not client:
            ai_text = "AI analysis not run: OPENAI_API_KEY is not configured."
            ai_error = "OPENAI_API_KEY missing"
        else:
            try:
                response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a fraud risk analyst."},
        {"role": "user", "content": prompt}
    ],
)

ai_text = response.choices[0].message.content
            except Exception as e:
                ai_text = f"AI analysis failed: {str(e)}"
                ai_error = str(e)

        # -------------------------
        # Email analyst
        # -------------------------
        email_sent = False
        email_error = None

        if not RESEND_API_KEY:
            email_error = "RESEND_API_KEY missing"
        else:
            try:
                resend.Emails.send(
                    {
                        # Works immediately in Resend without domain verification
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
                email_sent = True
            except Exception as e:
                email_error = str(e)
                print("Email send error:", e)

        return {
            "ok": True,
            "submission_id": submission_id,
            "message": "Submitted successfully.",
            "email_sent": email_sent,
            "email_error": email_error,
            "ai_error": ai_error,
        }

    except Exception as e:
        # This prevents "NetworkError" mystery failures and returns a real error
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )
