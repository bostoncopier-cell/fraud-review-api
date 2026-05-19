import os
import uuid
import base64
import asyncio
import sentry_sdk
import requests

from datetime import datetime
from typing import List, Tuple

from sentry_sdk.integrations.fastapi import FastApiIntegration

from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openai import OpenAI
import resend
from supabase import create_client


sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
    send_default_pii=False,
)


try:
    from pypdf import PdfReader

    PDF_TEXT_EXTRACTION = True
except Exception:
    PdfReader = None
    PDF_TEXT_EXTRACTION = False


app = FastAPI()

ALLOWED_ORIGINS = [
    "https://fraudreview.app",
    "https://www.fraudreview.app",
    "https://fraudreview-portal.vercel.app",
    "https://fraudreview-portal-4-16.vercel.app",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANALYST_EMAIL = "bostoncopier@gmail.com"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
resend.api_key = RESEND_API_KEY


@app.get("/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(OPENAI_API_KEY),
        "resend_configured": bool(RESEND_API_KEY),
        "pdf_text_extraction_enabled": PDF_TEXT_EXTRACTION,
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "sentry_configured": bool(os.environ.get("SENTRY_DSN")),
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
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(text)

        return "\n\n".join(text_parts).strip()[:limit_chars]

    except Exception as e:
        sentry_sdk.capture_exception(e)
        return ""


def _as_data_url(content_type: str, data: bytes) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


def _resend_attachments(files: List[Tuple[str, bytes]]) -> list:
    return [
        {
            "filename": filename,
            "content": base64.b64encode(data).decode("utf-8"),
        }
        for filename, data in files
    ]


def _build_ai_result(ai_text: str, ai_error: str | None = None) -> dict:
    return {
        "risk_level": "High"
        if "High" in ai_text
        else ("Moderate" if "Moderate" in ai_text else "Low"),
        "summary": ai_text,
        "reasoning_summary": ai_text,
        "signals_detected": [],
        "recommended_human_actions": [],
        "requires_escalation": "High" in ai_text or "escalate" in ai_text.lower(),
        "ai_error": ai_error,
    }


async def _update_submission_with_ai_result(
    submission_id: str,
    ai_result: dict,
):
    if not supabase:
        print("❌ Supabase not configured; cannot update background AI result")
        return

    for attempt in range(5):
        try:
            result = (
                supabase.table("submissions")
                .update(
                    {
                        "status": "awaiting_human_review",
                        "ai_result_json": ai_result,
                    }
                )
                .eq("reference_id", submission_id)
                .execute()
            )

            if getattr(result, "data", None):
                print(f"✅ Background AI result saved for {submission_id}")
                return

            print(f"⏳ Submission not found yet. Retry {attempt + 1}/5")

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print("❌ Supabase background update error:", e)

        await asyncio.sleep(1)

    print(f"⚠️ Background AI result could not be matched to {submission_id}")


async def process_submission_background(
    submission_id: str,
    transaction_type: str,
    contact_email: str,
    short_description: str,
    client_name_clean: str,
    raw_files: List[Tuple[str, bytes, str]],
):
    try:
        combined_text_chunks = []
        image_inputs = []

        for filename, data, ctype in raw_files:
            ctype_lower = (ctype or "").lower()

            if "pdf" in ctype_lower or filename.lower().endswith(".pdf"):
                pdf_text = _extract_pdf_text(data)

                if pdf_text.strip():
                    combined_text_chunks.append(
                        f"--- PDF TEXT ({filename}) ---\n{pdf_text}\n"
                    )
                else:
                    combined_text_chunks.append(
                        f"--- PDF ({filename}) ---\n(Unable to extract text reliably)\n"
                    )

            elif ctype_lower.startswith("image/") or filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp")
            ):
                image_inputs.append(
                    {
                        "type": "input_image",
                        "image_url": _as_data_url(
                            ctype_lower
                            if ctype_lower.startswith("image/")
                            else "image/png",
                            data,
                        ),
                    }
                )

            else:
                text = _safe_decode_text(data)

                if text.strip():
                    combined_text_chunks.append(
                        f"--- TEXT ({filename}) ---\n{text}\n"
                    )
                else:
                    combined_text_chunks.append(
                        f"--- FILE ({filename}) ---\n(Binary or unreadable as text)\n"
                    )

        combined_text = "\n\n".join(combined_text_chunks).strip()

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
If there is extracted text from a PDF, email, or document, use it too.

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

                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[{"role": "user", "content": content}],
                )

                ai_text = (
                    response.output_text.strip()
                    if getattr(response, "output_text", None)
                    else "(No AI output)"
                )

            except Exception as e:
                sentry_sdk.capture_exception(e)
                ai_text = "AI analysis failed."
                ai_error = str(e)

        ai_result = _build_ai_result(ai_text=ai_text, ai_error=ai_error)

        await _update_submission_with_ai_result(
            submission_id=submission_id,
            ai_result=ai_result,
        )

        if RESEND_API_KEY:
            try:
                attach_pairs = [(filename, data) for filename, data, _ctype in raw_files]

                resend.Emails.send(
                    {
                        "from": "Fraud Review <onboarding@resend.dev>",
                        "to": [ANALYST_EMAIL],
                        "subject": f"Fraud Review Submission {submission_id}",
                        "html": f"""
                            <h2>Fraud Review Submission</h2>
                            <p><b>Submission ID:</b> {submission_id}</p>
                            <p><b>Client / Person Name:</b> {client_name_clean if client_name_clean else "(not provided)"}</p>
                            <p><b>Transaction Type:</b> {transaction_type}</p>
                            <p><b>User Contact Email:</b> {contact_email}</p>
                            <p><b>Description:</b> {short_description}</p>
                            <p><b>Files attached:</b> {", ".join([fn for fn, _, _ in raw_files])}</p>
                            <hr/>
                            <pre style="white-space:pre-wrap;">{ai_text}</pre>
                        """,
                        "attachments": _resend_attachments(attach_pairs),
                    }
                )

                print(f"✅ Analyst email sent for {submission_id}")

            except Exception as e:
                sentry_sdk.capture_exception(e)
                print("❌ Email send error:", e)

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print("❌ Background processing error:", e)


@app.post("/api/submit")
async def submit(
    background_tasks: BackgroundTasks,
    transaction_type: str = Form(...),
    contact_email: str = Form(...),
    short_description: str = Form(""),
    client_name: str = Form(""),
    files: List[UploadFile] = File(...),
):
    try:
        submission_id = str(uuid.uuid4())
        client_name_clean = (client_name or "").strip()

        raw_files: List[Tuple[str, bytes, str]] = []

        for uploaded_file in files:
            data = await uploaded_file.read()

            raw_files.append(
                (
                    uploaded_file.filename or "upload",
                    data,
                    uploaded_file.content_type or "application/octet-stream",
                )
            )

        background_tasks.add_task(
            process_submission_background,
            submission_id,
            transaction_type,
            contact_email,
            short_description,
            client_name_clean,
            raw_files,
        )

        return {
            "ok": True,
            "submission_id": submission_id,
            "message": "Your submission was received and is now being prepared for fraud screening and human review.",
            "email_sent": False,
            "email_error": None,
            "ai_error": None,
            "files_received": [filename for filename, _, _ in raw_files],
            "client_name": client_name_clean,
            "ai_result": None,
        }

    except Exception as e:
        sentry_sdk.capture_exception(e)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Submission failed."},
        )


@app.post("/api/submit-uploadthing")
async def submit_uploadthing(
    background_tasks: BackgroundTasks,
    request: Request,
):
    try:
        body = await request.json()

        transaction_type = body.get("transaction_type", "")
        contact_email = body.get("contact_email", "")
        short_description = body.get("short_description", "")
        client_name_clean = (body.get("client_name", "") or "").strip()

        uploadthing_url = body.get("uploadthing_url", "")
        uploadthing_key = body.get("uploadthing_key", "")
        file_name = body.get("file_name", "upload")
        file_type = body.get("file_type", "application/octet-stream")

        if not transaction_type or not contact_email or not uploadthing_url:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": "Missing required UploadThing fields.",
                },
            )

        submission_id = str(uuid.uuid4())

        print("⬇️ Downloading UploadThing file:", uploadthing_url)

        download_response = await asyncio.to_thread(
            lambda: requests.get(uploadthing_url, timeout=30)
        )

        if download_response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": "Failed to download UploadThing file.",
                },
            )

        file_bytes = download_response.content

        raw_files = [
            (
                file_name,
                file_bytes,
                file_type,
            )
        ]

        background_tasks.add_task(
            process_submission_background,
            submission_id,
            transaction_type,
            contact_email,
            short_description,
            client_name_clean,
            raw_files,
        )

        return {
            "ok": True,
            "submission_id": submission_id,
            "message": "Your submission was received and is now being prepared for fraud screening and human review.",
            "email_sent": False,
            "email_error": None,
            "ai_error": None,
            "files_received": [file_name],
            "uploadthing_key": uploadthing_key,
            "client_name": client_name_clean,
            "ai_result": None,
        }

    except Exception as e:
        sentry_sdk.capture_exception(e)

        print("❌ UploadThing submission error:", e)

        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "UploadThing submission failed.",
            },
        )


@app.post("/api/inbound/resend")
async def inbound_email(request: Request):
    try:
        body = await request.json()

        print("🔥 INBOUND EMAIL RECEIVED")

        data = body.get("data", {})

        email_from = data.get("from", "")
        subject = data.get("subject", "")
        text = data.get("text", "")
        attachments = data.get("attachments", [])

        attachment_count = len(attachments)
        has_attachments = attachment_count > 0

        submission = {
            "source": "email",
            "contact_email": email_from,
            "email_from": email_from,
            "email_subject": subject,
            "email_body": text,
            "attachment_count": attachment_count,
            "has_attachments": has_attachments,
            "raw_email_json": data,
        }

        if not supabase:
            print("❌ Supabase not configured")
        else:
            supabase.table("submissions").insert(submission).execute()
            print("✅ Saved to Supabase")

        return {"status": "ok"}

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print("❌ Inbound email error:", e)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Inbound email processing failed."},
        )


@app.post("/api/submissions/{submission_id}/delete")
async def soft_delete_submission(submission_id: str):
    try:
        if not supabase:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": "Supabase not configured"},
            )

        deleted_at = datetime.utcnow().isoformat()

        result = (
            supabase.table("submissions")
            .update({"deleted_at": deleted_at})
            .eq("id", submission_id)
            .execute()
        )

        print("DELETE RESULT:", result)

        return {"ok": True, "message": "Submission removed from active view"}

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print("DELETE ERROR:", str(e))
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Delete failed."},
        )


@app.post("/api/submissions/{submission_id}/restore")
async def restore_submission(submission_id: str):
    try:
        if not supabase:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": "Supabase not configured"},
            )

        result = (
            supabase.table("submissions")
            .update({"deleted_at": None})
            .eq("id", submission_id)
            .execute()
        )

        print("RESTORE RESULT:", result)

        return {"ok": True, "message": "Submission restored to active view"}

    except Exception as e:
        sentry_sdk.capture_exception(e)
        print("RESTORE ERROR:", str(e))
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Restore failed."},
        )
