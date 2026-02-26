import os
import uuid
import base64
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openai import OpenAI
import resend

# Optional PDF text extraction
try:
    from pypdf import PdfReader  # pip install pypdf
    PDF_TEXT_EXTRACTION = True
except Exception:
    PdfReader = None
    PDF_TEXT_EXTRACTION = False


app = FastAPI()

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

# ✅ Send submissions to BOTH recipients
ANALYST_EMAILS = [
    "bostoncopier@gmail.com",
    "Larry@stirmgroup.com",
]

# Environment variables (set in Render -> Environment)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
resend.api_key = RESEND_API_KEY


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(OPENAI_API_KEY),
        "resend_configured": bool(RESEND_API_KEY),
        "pdf_text_extraction_enabled": PDF_TEXT_EXTRACTION,
    }


def _safe_decode_text(data: bytes, limit: int = 12000) -> str:
    try:
        return data.decode("utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def _extract_pdf_text(data: bytes, limit_chars: int = 20000) -> str:
    if not (PDF_TEXT_EXTRACTION and PdfReader):
        return ""
    try:
        import io
        reader = PdfReader(io.BytesIO(data))
        text_parts = []
        for page in reader.pages[:10]:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
        joined = "\n\n".join(text_parts).strip()
        return joined[:limit_chars]
    except Exception:
        return ""


def _as_data_url(content_type: str, data: bytes) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


def _resend_attachments(files: List[Tuple[str, bytes]]) -> list:
    """
    Resend expects attachments: [{"filename": "...", "content": "<base64>"}]
    """
    out = []
    for filename, data in files:
        out.append(
            {
                "filename": filename,
                "content": base64.b64encode(data).decode("utf-8"),
            }
        )
    return out


@app.post("/api/submit")
async def submit(
    transaction_type: str = Form(...),
    contact_email: str = Form(...),
    short_description: str = Form(""),
    client_name: str = Form(""),  # optional
    files: List[UploadFile] = File(...),
):
    try:
        submission_id = str(uuid.uuid4())
        client_name_clean = (client_name or "").strip()

        # Read all uploaded bytes (keep for attachments)
        raw_files: List[Tuple[str, bytes, str]] = []  # (filename, bytes, content_type)
        for f in files:
            data = await f.read()
            raw_files.append(
                (f.filename or "upload", data, f.content_type or "application/octet-stream")
            )

        # Build evidence text + image inputs for AI
        combined_text_chunks = []
        image_inputs = []

        for filename, data, ctype in raw_files:
            ctype_lower = (ctype or "").lower()

            # PDFs: try text extraction
            if "pdf" in ctype_lower or filename.lower().endswith(".pdf"):
                pdf_text = _extract_pdf_text(data)
                if pdf_text.strip():
                    combined_text_chunks.append(f"--- PDF TEXT ({filename}) ---\n{pdf_text}\n")
                else:
                    combined_text_chunks.append(f"--- PDF ({filename}) ---\n(Unable to extract text reliably)\n")

            # Images: send to vision model
            elif ctype_lower.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_inputs.append(
                    {
                        "type": "input_image",
                        "image_url": _as_data_url(
                            ctype_lower if ctype_lower.startswith("image/") else "image/png",
                            data
                        ),
                    }
                )

            # Email files / other text-like
            else:
                text = _safe_decode_text(data)
                if text.strip():
                    combined_text_chunks.append(f"--- TEXT ({filename}) ---\n{text}\n")
                else:
                    combined_text_chunks.append(f"--- FILE ({filename}) ---\n(Binary or unreadable as text)\n")

        combined_text = "\n\n".join(combined_text_chunks).strip()

        # -------------------------
        # AI analysis (Vision + Text)
        # -------------------------
        ai_text = ""
        ai_error = None

        if not client:
            ai_text = "AI analysis not run: OPENAI_API_KEY is not configured."
            ai_error = "OPENAI_API_KEY missing"
        else:
            try:
                prompt = f"""
You are a fraud risk analyst. Assess the submitted transaction communication for fraud risk.

Client / Person Name: {client_name_clean if client_name_clean else "(not provided)"}
Transaction Type: {transaction_type}
Description: {short_description}
User Contact Email: {contact_email}

If there are images, they may be screenshots of emails or wire instructions—read them carefully.
If there is extracted text (PDF/email), use it too.

Return exactly:
1) Risk Level: Low / Moderate / High
2) Key Findings (bullets)
3) Short Assessment (2-4 sentences)
4) Recommendation (bullets)
"""

                content = [{"type": "input_text", "text": prompt}]
                if combined_text:
                    content.append(
                        {
                            "type": "input_text",
                            "text": f"\n\nExtracted / forwarded text:\n{combined_text}",
                        }
                    )
                if image_inputs:
                    content.extend(image_inputs)

                resp = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[{"role": "user", "content": content}],
                )
                ai_text = resp.output_text.strip() if getattr(resp, "output_text", None) else "(No AI output)"
            except Exception as e:
                ai_text = f"AI analysis failed: {str(e)}"
                ai_error = str(e)

        # -------------------------
        # Email analyst (include attachments)
        # -------------------------
        email_sent = False
        email_error = None

        if not RESEND_API_KEY:
            email_error = "RESEND_API_KEY missing"
        else:
            try:
                attach_pairs = [(fn, data) for (fn, data, _ctype) in raw_files]

                resend.Emails.send(
                    {
                        "from": "Fraud Review <onboarding@resend.dev>",
                        "to": ANALYST_EMAILS,  # ✅ BOTH recipients
                        "subject": f"Fraud Review Submission {submission_id}",
                        "html": f"""
                            <h2>Fraud Review Submission</h2>
                            <p><b>Submission ID:</b> {submission_id}</p>
                            <p><b>Client / Person Name:</b> {client_name_clean if client_name_clean else "(not provided)"}</p>
                            <p><b>Transaction Type:</b> {transaction_type}</p>
                            <p><b>User Contact Email:</b> {contact_email}</p>
                            <p><b>Description:</b> {short_description}</p>
                            <p><b>Files attached:</b> {", ".join([fn for fn,_,_ in raw_files])}</p>
                            <hr/>
                            <pre style="white-space:pre-wrap;">{ai_text}</pre>
                        """,
                        "attachments": _resend_attachments(attach_pairs),
                    }
                )

                email_sent = True
            except Exception as e:
                email_error = str(e)
                print("Email send error:", e)

        # ✅ This is what your HTML will display as the “thank you”
        return {
            "ok": True,
            "submission_id": submission_id,
            "message": "Thank you — your submission has been received. An analyst will follow up with an independent advisory opinion shortly.",
            "email_sent": email_sent,
            "email_error": email_error,
            "ai_error": ai_error,
            "files_received": [fn for fn, _, _ in raw_files],
            "client_name": client_name_clean,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
