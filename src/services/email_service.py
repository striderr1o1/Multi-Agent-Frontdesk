import os
import httpx
from dotenv import load_dotenv
load_dotenv()

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"


async def send_email(to_email: str, subject: str, html_content: str) -> dict:
    """Send a transactional email through Brevo's REST API.

    Raises on any non-2xx response or network failure so callers can decide
    how to surface the error (e.g. wrap it in a ToolException).
    """
    api_key = os.environ.get("BREVO_API_KEY")
    sender_email = os.environ.get("BREVO_SENDER_EMAIL")
    sender_name = os.environ.get("BREVO_SENDER_NAME", "Operations Copilot")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            BREVO_API_URL,
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "sender": {"name": sender_name, "email": sender_email},
                "to": [{"email": to_email}],
                "subject": subject,
                "htmlContent": html_content,
            },
        )
    response.raise_for_status()
    return response.json()

def resolve_base_url():
    railway_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    if railway_domain:
        return f"https://{railway_domain}"
    return "http://localhost:8000"

API_BASE_URL = resolve_base_url()

def get_html_content(email: str, time_start: str, time_end: str, verification_id: str) -> str:
    """Build the HTML body for a booking verification email."""
    verification_url = f"{API_BASE_URL}/booking-confirmation/{verification_id}"
    return f"""
    <html>
        <body>
            <p>A booking was requested for <strong>{email}</strong>.</p>
            <p><strong>Start:</strong> {time_start}<br>
               <strong>End:</strong> {time_end}</p>
            <p><a href="{verification_url}">Click here to confirm your booking</a></p>
        </body>
    </html>
    """
