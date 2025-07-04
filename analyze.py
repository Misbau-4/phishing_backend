# analyze.py
from urllib.parse import urlparse
import tldextract
from bs4 import BeautifulSoup
import re
from difflib import SequenceMatcher
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import tldextract
from difflib import SequenceMatcher
import re

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def analyze_url_playwright(url: str) -> dict:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=15000)
        html = page.content()
        parsed = urlparse(url)
        domain = parsed.netloc
        tld = tldextract.extract(url).suffix
        is_https = parsed.scheme == 'https'
        
        soup = BeautifulSoup(html, 'html.parser')
        title_tag = soup.title.string.strip() if soup.title and soup.title.string else ''

        special_chars = re.findall(r'[^a-zA-Z0-9:/\.\-_%]', url)
        no_of_special = len(special_chars)
        special_ratio = no_of_special / max(len(url), 1)

        result = {
            'URL': url,
            'Domain': domain,
            'TLD': tld,
            'URLSimilarityIndex': similarity(url, title_tag),
            'NoOfOtherSpecialCharsInURL': no_of_special,
            'SpacialCharRatioInURL': special_ratio,
            'IsHTTPS': int(is_https),
            'LineOfCode': html.count('\n') + 1,
            'Title': title_tag,
            'DomainTitleMatchScore': similarity(domain, title_tag),
            'URLTitleMatchScore': similarity(parsed.path, title_tag),
            'IsResponsive': int(bool(soup.find('meta', attrs={'name': 'viewport'}))),
            'HasDescription': int(bool(soup.find('meta', attrs={'name': 'description'}))),
            'HasSocialNet': int(bool(soup.find(lambda tag: tag.name == 'a' and any(net in tag.get('href', '') for net in ('facebook.com', 'twitter.com', 'instagram.com'))))),
            'HasSubmitButton': int(bool(soup.select_one('input[type="submit"], button[type="submit"]'))),
            'HasCopyrightInfo': int(bool(re.search(r'©|&copy;', html))),
            'NoOfImage': len(soup.find_all('img')),
            'NoOfJS': len(soup.find_all('script')),
            'NoOfSelfRef': len([a for a in soup.find_all('a', href=True) if domain in a['href']])
        }

        browser.close()
        return result
