from email import policy
from email.parser import BytesParser
from pathlib import Path

from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm


def text_from_html(text: str) -> str:
    try:
        text = " ".join(BeautifulSoup(text, "html.parser").stripped_strings)
    except Exception:
        pass
    return text


def parse_email_file(file_path: Path) -> dict:
    try:
        raw_data = file_path.read_bytes()
        msg = BytesParser(policy=policy.default).parsebytes(raw_data)  # type: ignore[arg-type]

        decoded_content = None

        text_parts = []
        html_parts = []

        for part in msg.walk():
            content_type = part.get_content_type()

            if content_type in ["text/plain", "text/html"]:
                payload = part.get_payload(decode=True)
                if not payload:
                    continue

                charset = part.get_content_charset() or "utf-8"

                try:
                    decoded_content = payload.decode(charset, errors="replace")  # type: ignore[union-attr]
                except LookupError:
                    decoded_content = payload.decode("utf-8", errors="replace")  # type: ignore[union-attr]

                if content_type == "text/plain":
                    if any(tag in decoded_content.lower() for tag in ["<html", "<body", "<div", "<p>", "href=", "<br"]):
                        html_parts.append(text_from_html(decoded_content))
                    else:
                        text_parts.append(decoded_content.strip())
                elif content_type == "text/html":
                    html_parts.append(text_from_html(decoded_content))

        return {
            "file_path": str(file_path),
            "subject": msg.get("subject", ""),
            "from": msg.get("from", ""),
            "to": msg.get("to", ""),
            "date": msg.get("date", ""),
            "content_type": msg.get_content_type(),
            "content": "\n".join(text_parts).strip(),
            "html_content": "\n".join(html_parts).strip(),
            "is_multipart": msg.is_multipart(),
            "num_attachments": sum(1 for part in msg.iter_attachments() if part.get_filename()),  # type: ignore[misc]
        }
    except Exception as e:
        logger.error(f"Parsing error file {file_path.name}: {e}")
        return {
            "file_path": str(file_path),
            "error": str(e),
            "content": file_path.read_text(errors="replace")[:1000] + "... [TRUNCATED]",
        }


def get_parsed_data(parse_dir: Path, label: int) -> list[dict]:
    data = []
    files = [f for f in parse_dir.rglob("*") if f.is_file() and f.stat().st_size > 0]

    for file_path in tqdm(files, desc="Processing emails"):
        parsed = parse_email_file(file_path)
        parsed["label"] = label
        data.append(parsed)
    return data
