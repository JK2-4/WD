#!/usr/bin/env python3
"""
fetch_max_hjb_content.py

  - the full HTML
  - its plain‑text content
  - every <code> block (e.g. code listings)
  - every LaTeX/math expression (in $$…$$ or \\[…\\])
"""

import requests
from bs4 import BeautifulSoup
import re
import os

URL = "https://s3.amazonaws.com/rendezvouswithdestiny.me/finance/max_hjb_mean_var.html"
OUT_DIR = "max_hjb_content"

def ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

def download_html(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.text

def save_file(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def extract_text(soup):
    # collapse multiple blank lines
    text = soup.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()

def extract_code_blocks(soup):
    return [code.get_text() for code in soup.find_all("code")]

def extract_math(html):
    # find $$...$$ and \[...\] blocks
    blocks = re.findall(r"\$\$(.+?)\$\$", html, flags=re.DOTALL)
    blocks += re.findall(r"\\\[(.+?)\\\]", html, flags=re.DOTALL)
    return [blk.strip() for blk in blocks]

def main():
    ensure_dir(OUT_DIR)
    html = download_html(URL)
    save_file(f"{OUT_DIR}/max_hjb_mean_var.html", html)
    soup = BeautifulSoup(html, "html.parser")

    # 1. Plain text
    text = extract_text(soup)
    save_file(f"{OUT_DIR}/max_hjb_text.txt", text)

    # 2. Code blocks
    codes = extract_code_blocks(soup)
    with open(f"{OUT_DIR}/max_hjb_code_blocks.txt", "w", encoding="utf-8") as f:
        for i, blk in enumerate(codes, 1):
            f.write(f"--- Code block {i} ---\n")
            f.write(blk + "\n\n")

    # 3. Math/LaTeX blocks
    maths = extract_math(html)
    with open(f"{OUT_DIR}/max_hjb_math_eqns.txt", "w", encoding="utf-8") as f:
        for i, eq in enumerate(maths, 1):
            f.write(f"--- Equation {i} ---\n")
            f.write(eq + "\n\n")

    print(f"All content saved under ./{OUT_DIR}/")

if __name__ == "__main__":
    main()
