# ruff: noqa: T201
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "beautifulsoup4",
#     "requests",
# ]
# ///
import re
from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup  # type: ignore


def extract_response_text(response: requests.Response) -> str:
    """Extract response as plain text and clean up."""
    soup = BeautifulSoup(response.text, "html.parser")
    plain_text = soup.get_text()

    # Collapse multiple blank lines.
    clean_text = re.sub(r"\n\s*\n+", "\n\n", plain_text.strip())

    return clean_text


def format_email(info: dict[str, Any]) -> str:
    headers = "\n".join(
        f"  {key}: {value}" for key, value in info["request_headers"].items()
    )

    return f"""\
Dear census.gov support team,

I'm encountering a 403 HTTPError when attempting to download a file from the
www2.census.gov domain using Python (requests).

Below is the diagnostic information:

URL Accessed: {info["url"]}

Time of Error: {info["timestamp"]}

User Agent: {info["user_agent"]}

Full Request Headers:

{headers}

Response Content:

{info["response_text"] or "(no HTML response detected)"}

Thank you for your assistance.

Best regards,
[Your Name]
"""


def check(url: str) -> None:
    timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    response = requests.get(url, timeout=60)

    if response.status_code == 200:
        print("No error, file access is good.")
        # print(response.headers)
    else:
        message = format_email(
            {
                "timestamp": timestamp,
                "url": response.request.url,
                "user_agent": response.request.headers["User-Agent"],
                "error_message": f"HTTP {response.status_code}: {response.reason}",
                "response_text": extract_response_text(response),
                "request_headers": response.request.headers,
            }
        )
        print(message)


if __name__ == "__main__":
    tiger_url = "https://www2.census.gov/geo/tiger"
    state = "04"
    url_2020 = f"{tiger_url}/TIGER2020/TRACT/tl_2020_{state}_tract.zip"
    check(url_2020)
