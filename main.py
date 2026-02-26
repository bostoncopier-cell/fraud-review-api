import os
import uuid
import base64
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openai import OpenAI
import resend

# Optional: for text-based PDFs (pure python)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


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

# Limits (keep sane for free-tier + email size limits)
MAX_FILES = 5
MAX_FILE_BYTES = 6 * 1024 * 1024  # 6MB each
MAX_TEXT_CHARS = 12000


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(OPENAI_API_KEY),
        "resend_configured": bool(RESEND_API_KEY),
        "pdf_text_extraction_enabled": bool(PdfReader),
    }


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _guess_mime(filename: str, content_type: str | None) -> str:
    if content_type:
        return content_type
    fn = (filename or "").lower()
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".jpg") or fn.endswith(".jpeg"):
        return "image/jpeg"
    if fn.endswith(".pdf"):
        return "application/pdf"
    return "application/octet-stream"


def _is_image(mime: str) -> bool:
    return mime in ("image/png", "image/jpeg")


def _is_pdf(mime: str) -> bool:
    return mime == "application/pdf"


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extracts text from text-based PDFs only.
    If the PDF is scanned (image-only), this will return empty/near-empty.
    """
    if not PdfReader:
        return ""

    try:
        import io
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages[:10]:  # cap pages
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt.strip())
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def _build_prompt(transaction_type: str, short_description: str, contact_email: str) -> str:
    return f"""
You are a fraud-risk analyst. Analyze the uploaded email / wire instructions / screenshots for fraud risk.

Transaction Type: {transaction_type}
User Description: {short_description}
User Contact Email: {contact_email}

Return ONLY this structure:

Risk Level: Low | Moderate | High
Key Findings:
- bullet
- bullet
Short Assessment: 2-5 sentences
Recommendation: 2-5 bullets

Be specific about red flags (urgency, sender spoofing, payment method, domain mismatch, changed wire instructions, odd phrasing, attachments, request to bypass procedure, etc.).
""".strip()


@app.post("/api/submit")
async def submit(
    transaction_type: str = Form(...),
    contact_email: str = Form(...),
    short_description: str = Form(""),
    files: List[UploadFile] = File(...),
):
    try:
        submission_id = str(uuid.uuid4())

        if not files or len(files) == 0:
            return JSONResponse(status_code=400, content={"ok": False, "error": "No files uploaded."})

        if len(files) > MAX_FILES:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"Too many files. Max allowed is {MAX_FILES}."},
            )

        # Read & validate files
        uploaded: List[Tuple[str, str, bytes]] = []  # (filename, mime, bytes)
        for f in files:
            data = await f.read()
            if not data:
                continue

            if len(data) > MAX_FILE_BYTES:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": f"File too large: {f.filename}. Max is {MAX_FILE_BYTES} bytes."},
                )

            mime = _guess_mime(f.filename, f.content_type)
            uploaded.append((f.filename or "upload", mime, data))

        if not uploaded:
            return JSONResponse(status_code=400, content={"ok": False, "error": "Uploaded files were empty."})

        prompt = _build_prompt(transaction_type, short_description, contact_email)

        # -------------------------
        # AI analysis (images via vision; PDFs via text-extract)
        # -------------------------
        ai_text = ""
        ai_error = None

        if not client:
            ai_text = "AI analysis not run: OPENAI_API_KEY is not configured."
            ai_error = "OPENAI_API_KEY missing"
        else:
            try:
                # Build one message with text + (optional) multiple images
                content_parts = [{"type": "text", "text": prompt}]

                # Accumulate extracted text from text-based files
                extracted_text_blocks = []

                for (filename, mime, data) in uploaded:
                    if _is_image(mime):
                        data_url = f"data:{mime};base64,{_b64(data)}"
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": data_url}}
                        )
                    elif _is_pdf(mime):
                        pdf_text = _extract_pdf_text(data)
                        if pdf_text:
                            extracted_text_blocks.append(
                                f"--- PDF TEXT EXTRACT ({filename}) ---\n{pdf_text}"
                            )
                        else:
                            extracted_text_blocks.append(
                                f"--- PDF NOTE ({filename}) ---\n"
                                f"This PDF appears to have no extractable text (likely scanned/image-only). "
                                f"For best results, upload a screenshot image (JPG/PNG) of the email/wire instructions."
                            )
                    else:
                        # Try a best-effort decode for .eml/.msg/.txt-like uploads
                        try:
                            text = data.decode("utf-8", errors="ignore").strip()
                            if text:
                                extracted_text_blocks.append(f"--- FILE TEXT ({filename}) ---\n{text[:MAX_TEXT_CHARS]}")
                        except Exception:
                            pass

                if extracted_text_blocks:
                    content_parts.append(
                        {"type": "text", "text": "\n\n".join(extracted_text_blocks)[:MAX_TEXT_CHARS]}
                    )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a careful, professional fraud risk analyst."},
                        {"role": "user", "content": content_parts},
                    ],
                )

                ai_text = (response.choices[0].message.content or "").strip()
                if not ai_text:
                    ai_text = "AI returned an empty analysis."
            except Exception as e:
                ai_text = f"AI analysis failed: {str(e)}"
                ai_error = str(e)

        # -------------------------
        # Email analyst + attach originals
        # -------------------------
        email_sent = False
        email_error = None

        attachments = []
        for (filename, mime, data) in uploaded:
            # Resend attachments want base64 content (most libs accept "content" base64)
            attachments.append(
                {
                    "filename": filename,
                    "content": _b64(data),
                    "content_type": mime,
                }
            )

        if not RESEND_API_KEY:
            email_error = "RESEND_API_KEY missing"
        else:
            try:
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
                            <p><b>Files:</b> {", ".join([u[0] for u in uploaded])}</p>
                            <hr/>
                            <pre style="white-space:pre-wrap;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">{ai_text}</pre>
                        """,
                        "attachments": attachments,
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
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
