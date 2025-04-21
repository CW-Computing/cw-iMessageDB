import sqlite3
import plistlib
import os
import re
import sys
from datetime import datetime

# --- Color Codes ---
COLOR_ME = '\033[92m'    # Green
COLOR_THEM = '\033[96m'  # Cyan
COLOR_RESET = '\033[0m'  # Reset

# --- Detect if output supports colors ---
USE_COLOR = sys.stdout.isatty()

def extract_clean_text_from_blob(blob):
    try:
        raw = blob.decode('utf-8', errors='ignore')
        if 'NSString' in raw:
            nsstring_index = raw.find('NSString')
            raw_after_nsstring = raw[nsstring_index:]
            plus_index = raw_after_nsstring.find('+')
            if plus_index != -1:
                text_start = plus_index + 1
                text_candidate = raw_after_nsstring[text_start:]
                for cutoff in ['\\x02', '\\x0c', '\\n', 'NSDictionary', 'NSValue']:
                    cut_idx = text_candidate.find(cutoff)
                    if cut_idx != -1:
                        text_candidate = text_candidate[:cut_idx]
                return text_candidate.strip()
        return ''.join(c for c in raw if 32 <= ord(c) <= 126).strip()
    except Exception as e:
        return f"[Error extracting text: {e}]"

def fully_clean_text(text):
    # 1. Remove all control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # 2. Remove trailing weird patterns like iI, iIA, iI<, etc
    text = re.sub(r'(\s*iI[a-zA-Z0-9]*\s*$)', '', text)

    # 3. Strip extra spaces
    return text.strip()

# --- Configuration ---
CHAT_DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")
TARGET_CONTACT = "+15555551212"  # <-- Replace with the number/email you want
LIMIT_MESSAGES = 50

# --- Connect to Database ---
print(f"Connecting to: {CHAT_DB_PATH}")
conn = sqlite3.connect(CHAT_DB_PATH)
cursor = conn.cursor()

# --- Query Messages ---
query = f"""
SELECT 
  message.date,
  handle.id AS contact,
  message.is_from_me,
  message.text,
  message.attributedBody
FROM 
  message
LEFT JOIN 
  handle ON message.handle_id = handle.rowid
WHERE
  (handle.id = '{TARGET_CONTACT}' OR message.is_from_me = 1)
  AND message.attributedBody IS NOT NULL
ORDER BY
  message.date
LIMIT {LIMIT_MESSAGES};
"""

cursor.execute(query)
messages = cursor.fetchall()
print(f"Fetched {len(messages)} messages with attributedBody for {TARGET_CONTACT}.")

# --- Output Messages ---
for msg in messages:
    date_val, contact, is_from_me, text_field, attributed_body_blob = msg

    # Convert Apple's weird date format
    if date_val:
        timestamp = int(date_val) / 1000000000 + datetime(2001, 1, 1).timestamp()
        message_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    else:
        message_date = ""

    # Try to decode attributedBody
    clean_text = extract_clean_text_from_blob(attributed_body_blob)

    # Fallback if somehow the text field was populated
    if not clean_text and text_field:
        clean_text = text_field

    # --- Deep clean text ---
    clean_text = fully_clean_text(clean_text)

    # Pick color or plain text depending on environment
    if is_from_me == 1:
        sender = f"{COLOR_ME}Me{COLOR_RESET}" if USE_COLOR else "Me"
    else:
        sender = f"{COLOR_THEM}{TARGET_CONTACT}{COLOR_RESET}" if USE_COLOR else TARGET_CONTACT

    # Print final output
    print(f"[{message_date}] {sender}: {clean_text}")

# --- Cleanup ---
cursor.close()
conn.close()

print("Done! ✅")