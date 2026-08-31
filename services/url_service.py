"""
URL Article Verification Service for TruthCheck
Extracts article content, cleans HTML/ads, assesses domain reliability, and fact-checks content.

NOTE: newspaper3k/newspaper4k is an optional dependency.
If not installed, content extraction falls back to requests + BeautifulSoup.
"""
import logging
from urllib.parse import urlparse
from services.groq_service import analyze_url_content

logger = logging.getLogger(__name__)

RELIABLE_DOMAINS = {
    'reuters.com': 96, 'bbc.com': 94, 'apnews.com': 97, 'bloomberg.com': 92,
    'nature.com': 98, 'wsj.com': 90, 'theguardian.com': 88, 'nytimes.com': 89
}

UNRELIABLE_DOMAINS = {
    'theonion.com': 15, 'fakenews.com': 5, 'infowars.com': 10
}


def _extract_with_requests(url):
    """Fallback content extractor using requests + BeautifulSoup."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {'User-Agent': 'Mozilla/5.0 (compatible; TruthCheck/2.0)'}
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else 'Untitled Article'

        # Extract body text — prefer article tags, fall back to paragraphs
        article_tag = soup.find('article')
        if article_tag:
            paragraphs = article_tag.find_all('p')
        else:
            paragraphs = soup.find_all('p')

        text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        if not text:
            text = soup.get_text(separator=' ', strip=True)[:5000]

        return {
            'title': title,
            'text': text[:5000],
            'authors': [],
            'publish_date': 'Unknown',
        }
    except Exception as exc:
        logger.warning("requests/BS4 fallback extraction failed: %s", exc)
        return {
            'title': 'Extraction Failed',
            'text': f'Could not extract article content: {exc}',
            'authors': [],
            'publish_date': 'Unknown',
        }


def extract_article(url):
    """Scrape article title and clean text from web URL.

    Tries newspaper3k first, falls back to requests + BeautifulSoup if unavailable.
    """
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        return {
            'title': article.title or "Untitled Article",
            'text': article.text or "",
            'authors': article.authors,
            'publish_date': str(article.publish_date) if article.publish_date else "Unknown"
        }
    except ImportError:
        logger.info("newspaper not installed — using requests/BeautifulSoup fallback.")
        return _extract_with_requests(url)
    except Exception as exc:
        logger.warning("newspaper extraction failed: %s — using fallback.", exc)
        return _extract_with_requests(url)


def assess_domain_reliability(domain):
    """Evaluate domain credibility against verified index."""
    domain_clean = domain.lower().replace('www.', '')
    for d, score in RELIABLE_DOMAINS.items():
        if d in domain_clean:
            return {'score': score, 'status': 'Verified Reliable Source'}
    for d, score in UNRELIABLE_DOMAINS.items():
        if d in domain_clean:
            return {'score': score, 'status': 'Flagged Unreliable / Satire'}
    return {'score': 65, 'status': 'Unrated Domain / General Web'}


def analyze_url(url, response_lang='en'):
    """Full URL verification pipeline."""
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    extracted = extract_article(url)
    domain_eval = assess_domain_reliability(domain)

    result = analyze_url_content(extracted['title'], extracted['text'], domain, response_lang=response_lang)
    if result:
        result['url'] = url
        result['domain'] = domain
        result['domain_reliability'] = domain_eval
        result['extracted_title'] = extracted['title']
        result['publish_date'] = extracted.get('publish_date')
    return result
