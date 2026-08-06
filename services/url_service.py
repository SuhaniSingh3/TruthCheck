"""
URL Article Verification Service for TruthCheck
Extracts article content, cleans HTML/ads, assesses domain reliability, and fact-checks content.
"""
from urllib.parse import urlparse
from newspaper import Article
from services.groq_service import analyze_url_content

RELIABLE_DOMAINS = {
    'reuters.com': 96, 'bbc.com': 94, 'apnews.com': 97, 'bloomberg.com': 92,
    'nature.com': 98, 'wsj.com': 90, 'theguardian.com': 88, 'nytimes.com': 89
}

UNRELIABLE_DOMAINS = {
    'theonion.com': 15, 'fakenews.com': 5, 'infowars.com': 10
}

def extract_article(url):
    """Scrape article title and clean text from web URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return {
            'title': article.title or "Untitled Article",
            'text': article.text or "",
            'authors': article.authors,
            'publish_date': str(article.publish_date) if article.publish_date else "Unknown"
        }
    except Exception as e:
        return {'title': 'Scraping Fallback', 'text': f'Could not directly scrape full text: {e}'}

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
