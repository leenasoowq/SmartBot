import pdfplumber
import requests
from bs4 import BeautifulSoup

# Extract text from a PDF file
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Extract text from a plain text file
def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""

# Extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        print(f"Error fetching URL content: {e}")
        return ""
