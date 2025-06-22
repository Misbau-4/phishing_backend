# analyze.py
import requests
from urllib.parse import urlparse
import tldextract
from bs4 import BeautifulSoup
import re
from difflib import SequenceMatcher

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def analyze_url(url: str) -> dict:
    parsed = urlparse(url)
    domain = parsed.netloc
    tld = tldextract.extract(url).suffix
    is_https = parsed.scheme == 'https'

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch URL: {e}")

    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.title.string.strip() if soup.title and soup.title.string else ''

    special_chars = re.findall(r'[^a-zA-Z0-9:/\.\-_%]', url)
    no_of_special = len(special_chars)
    special_ratio = no_of_special / max(len(url), 1)

    return {
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
