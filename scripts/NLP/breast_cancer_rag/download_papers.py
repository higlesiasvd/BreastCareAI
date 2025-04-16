import requests
from bs4 import BeautifulSoup
import time
import os
import random
import re
from urllib.parse import urljoin, quote
import json
from datetime import datetime

# Configuration
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5,en;q=0.3',
    'Referer': 'https://www.google.com/'
}
OUTPUT_DIR = 'guidelines_to_review'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# JSON file to store metadata
METADATA_FILE = os.path.join(OUTPUT_DIR, 'guideline_metadata.json')

# List of keywords to filter relevant guidelines and information resources
# Focus on patient information and clinical guidelines
KEYWORDS = [
    'breast cancer', 'breast carcinoma', 'mammary carcinoma',
    'patient guide', 'patient information', 'patient education',
    'guidelines', 'clinical pathway', 'recommendations',
    'treatment guidelines', 'standard of care', 'best practice',
    'clinical protocol', 'consensus statement', 'practice guideline',
    'patient resources', 'information leaflet', 'patient handbook',
    'care pathway', 'clinical guidance', 'patient decision aid',
    'information booklet', 'treatment algorithm', 'fact sheet'
]

# Guideline indicators to ensure we're getting proper guides
GUIDELINE_INDICATORS = [
    'guideline', 'guide', 'protocol', 'pathway', 'algorithm',
    'recommendation', 'consensus', 'standard of care', 'information',
    'booklet', 'handbook', 'factsheet', 'fact sheet', 'leaflet',
    'resource', 'patient education', 'decision aid', 'best practice'
]

def is_relevant_guideline(url, title="", abstract=""):
    """Check if a PDF is a relevant guideline or information resource based on keywords"""
    text_to_check = (url + " " + title + " " + abstract).lower()
    
    # Check if it contains at least one keyword
    keyword_match = any(keyword.lower() in text_to_check for keyword in KEYWORDS)
    if not keyword_match:
        return False
    
    # Check if it has guideline indicators
    guideline_match = any(indicator.lower() in text_to_check for indicator in GUIDELINE_INDICATORS)
    if not guideline_match:
        return False
    
    # Additional check for breast cancer specificity
    if not ('breast' in text_to_check and ('cancer' in text_to_check or 'carcinoma' in text_to_check or 'neoplasm' in text_to_check)):
        return False
        
    return True

def format_filename(s):
    """Convert a string to a safe filename"""
    # Replace non-alphanumeric characters with underscores
    s = re.sub(r'[^\w\s-]', '_', s)
    # Replace spaces with hyphens
    s = re.sub(r'\s+', '-', s)
    # Limit filename length
    return s[:100]

def check_pdf_validity(url):
    """Check if a URL points to a valid PDF"""
    try:
        response = requests.head(url, headers=HEADERS, timeout=10)
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Verify if it's a PDF by content type
        if 'application/pdf' in content_type:
            # Verify the size to ensure it's not too small (which could indicate an error)
            content_length = int(response.headers.get('Content-Length', '0'))
            if content_length > 10000:  # More than 10KB
                return True
        return False
    except Exception as e:
        print(f"Error verifying PDF {url}: {e}")
        return False

def search_pubmed_guidelines():
    """Search for breast cancer guidelines in PubMed"""
    base_url = "https://pubmed.ncbi.nlm.nih.gov"
    
    search_queries = [
        "breast+cancer+guideline+patient",
        "breast+cancer+patient+guide",
        "breast+cancer+clinical+practice+guideline",
        "breast+cancer+consensus+statement",
        "breast+cancer+patient+information+resource",
        "breast+cancer+standard+of+care",
        "breast+cancer+patient+decision+aid",
        "breast+cancer+recommendations+clinical",
        "breast+cancer+patient+education+material"
    ]
    
    print("Searching for guidelines in PubMed")
    
    pdf_links = []
    for query in search_queries:
        try:
            # Add publication type filter for practice guidelines
            search_url = f"{base_url}//?term={query}+AND+%28practice+guideline%5BPublication+Type%5D+OR+guideline%5BPublication+Type%5D%29&filter=simsearch1.fha&filter=years.2019-2024&size=100"
            
            response = requests.get(search_url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article items
                articles = soup.select("article.full-docsum")
                
                for article in articles:
                    # Get the article ID
                    article_id = article.get('data-article-id', '')
                    
                    if not article_id:
                        continue
                    
                    # Get title
                    title_elem = article.select_one("a.docsum-title")
                    title = title_elem.text.strip() if title_elem else ""
                    
                    # Get abstract URL
                    article_url = f"{base_url}/{article_id}/"
                    
                    # Get journal info
                    journal_elem = article.select_one(".docsum-journal-citation")
                    journal = journal_elem.text.strip() if journal_elem else ""
                    
                    # Fetch abstract
                    try:
                        article_response = requests.get(article_url, headers=HEADERS, timeout=15)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.content, 'html.parser')
                            
                            # Get abstract
                            abstract_elem = article_soup.select_one(".abstract-content")
                            abstract = abstract_elem.text.strip() if abstract_elem else ""
                            
                            # Look for full text links
                            full_text_links = []
                            for link in article_soup.select(".full-text-links-list a"):
                                full_text_links.append(link.get('href', ''))
                            
                            # Try to find the DOI
                            doi = ""
                            doi_elem = article_soup.select_one(".identifier.doi")
                            if doi_elem:
                                doi = doi_elem.text.strip().replace("DOI:", "").strip()
                            
                            # Check if this is a guideline publication type
                            is_guideline = False
                            publication_types = article_soup.select(".publication-type")
                            for pub_type in publication_types:
                                if "guideline" in pub_type.text.lower() or "practice guideline" in pub_type.text.lower():
                                    is_guideline = True
                                    break
                            
                            if is_relevant_guideline(article_url, title, abstract) or is_guideline:
                                # Add to list even without PDF link - we'll try to find it later
                                pdf_links.append({
                                    'url': article_url,
                                    'title': title,
                                    'abstract': abstract,
                                    'journal': journal,
                                    'doi': doi,
                                    'full_text_links': full_text_links,
                                    'source': 'PubMed',
                                    'pubmed_id': article_id,
                                    'language': 'en',
                                    'resource_type': 'Clinical Guideline' if is_guideline else 'Information Resource'
                                })
                    except Exception as e:
                        print(f"Error fetching article details: {e}")
            
            time.sleep(random.uniform(3, 5))
        except Exception as e:
            print(f"Error in PubMed search ({query}): {e}")
    
    print(f"Found {len(pdf_links)} relevant guidelines in PubMed")
    return pdf_links

def search_google_scholar_guidelines():
    """Search for breast cancer guidelines in Google Scholar"""
    base_url = "https://scholar.google.com/scholar"
    
    search_queries = [
        "breast cancer guideline patient information filetype:pdf",
        "breast cancer patient guide filetype:pdf",
        "breast cancer clinical practice guideline filetype:pdf",
        "breast cancer patient decision aid filetype:pdf",
        "breast cancer patient information booklet filetype:pdf",
        "breast cancer treatment algorithm filetype:pdf",
        "breast cancer standard of care guideline filetype:pdf",
        "breast cancer patient handbook filetype:pdf",
        "breast cancer care pathway filetype:pdf"
    ]
    
    print("Searching for guidelines in Google Scholar")
    
    pdf_links = []
    for query in search_queries:
        try:
            # Encode the query for URL
            encoded_query = quote(query)
            search_url = f"{base_url}?q={encoded_query}&hl=en&as_sdt=0,5&as_ylo=2019&num=30"
            
            response = requests.get(search_url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article items
                articles = soup.select(".gs_r.gs_or.gs_scl")
                
                for article in articles:
                    # Get title and main link
                    title_elem = article.select_one(".gs_rt a")
                    if not title_elem:
                        continue
                    
                    title = title_elem.text.strip()
                    article_url = title_elem.get('href', '')
                    
                    # Look for direct PDF links
                    pdf_link = None
                    for link in article.select(".gs_or_ggsm a"):
                        link_text = link.text.strip().lower()
                        link_url = link.get('href', '')
                        if "[pdf]" in link_text or link_url.lower().endswith('.pdf'):
                            pdf_link = link_url
                            break
                    
                    # Get abstract
                    abstract_elem = article.select_one(".gs_rs")
                    abstract = abstract_elem.text.strip() if abstract_elem else ""
                    
                    # Get publication info
                    pub_info_elem = article.select_one(".gs_a")
                    pub_info = pub_info_elem.text.strip() if pub_info_elem else ""
                    
                    # Try to extract journal/organization name from pub info
                    publisher = ""
                    if pub_info:
                        # Try to identify if this is from a guideline organization
                        guideline_orgs = [
                            "NCCN", "National Comprehensive Cancer Network", 
                            "ASCO", "American Society of Clinical Oncology",
                            "ESMO", "European Society for Medical Oncology",
                            "NICE", "National Institute for Health and Care Excellence",
                            "ACS", "American Cancer Society",
                            "Susan G. Komen", "Cancer Research UK",
                            "WHO", "World Health Organization"
                        ]
                        
                        for org in guideline_orgs:
                            if org.lower() in pub_info.lower():
                                publisher = org
                                break
                        
                        # If no specific org found, use the general info
                        if not publisher:
                            parts = pub_info.split('-')
                            if len(parts) > 1:
                                publisher = parts[1].strip()
                    
                    # Determine resource type based on title and abstract
                    resource_type = "Unknown"
                    if any(term in title.lower() or term in abstract.lower() for term in ["guideline", "guidance", "consensus"]):
                        resource_type = "Clinical Guideline"
                    elif any(term in title.lower() or term in abstract.lower() for term in ["patient guide", "handbook", "information", "booklet", "leaflet"]):
                        resource_type = "Patient Information"
                    elif any(term in title.lower() or term in abstract.lower() for term in ["algorithm", "pathway", "protocol", "decision aid"]):
                        resource_type = "Clinical Tool"
                    
                    if pdf_link and is_relevant_guideline(article_url, title, abstract):
                        pdf_links.append({
                            'url': pdf_link,
                            'title': title,
                            'abstract': abstract,
                            'publisher': publisher,
                            'pub_info': pub_info,
                            'article_url': article_url,
                            'source': 'Google Scholar',
                            'language': 'en',
                            'resource_type': resource_type
                        })
            
            # Important to wait between searches to avoid blocks
            time.sleep(random.uniform(10, 15))
        except Exception as e:
            print(f"Error in Google Scholar search ({query}): {e}")
    
    print(f"Found {len(pdf_links)} guidelines in Google Scholar")
    return pdf_links

def search_core_papers():
    """Search for breast cancer papers in CORE.ac.uk API"""
    # Note: CORE requires an API key for full functionality
    # This is a simplified version using their public search
    
    base_url = "https://core.ac.uk/search-results"
    
    search_queries = [
        "breast cancer patient education",
        "breast cancer treatment options",
        "breast cancer quality of life",
        "breast cancer decision making",
        "breast cancer survivorship"
    ]
    
    print("Searching for papers in CORE.ac.uk")
    
    pdf_links = []
    for query in search_queries:
        try:
            encoded_query = quote(query)
            search_url = f"{base_url}?q={encoded_query}&page=1&pageSize=100"
            
            response = requests.get(search_url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article listings
                articles = soup.select(".search-result-item")
                
                for article in articles:
                    # Get title
                    title_elem = article.select_one(".title")
                    if not title_elem:
                        continue
                    
                    title = title_elem.text.strip()
                    
                    # Get link to article page
                    link_elem = title_elem.find('a')
                    article_url = ""
                    if link_elem:
                        href = link_elem.get('href', '')
                        article_url = urljoin(base_url, href) if href else ""
                    
                    # Get abstract
                    abstract_elem = article.select_one(".description")
                    abstract = abstract_elem.text.strip() if abstract_elem else ""
                    
                    # Get authors/journal info
                    authors_elem = article.select_one(".authors")
                    authors = authors_elem.text.strip() if authors_elem else ""
                    
                    # Only proceed if we have an article URL and it looks relevant
                    if article_url and is_relevant_paper(article_url, title, abstract):
                        # For each relevant article, visit the article page to look for PDF
                        try:
                            article_response = requests.get(article_url, headers=HEADERS, timeout=15)
                            if article_response.status_code == 200:
                                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                                
                                # Look for PDF download button
                                pdf_link = None
                                for link in article_soup.select("a"):
                                    link_text = link.text.strip().lower()
                                    href = link.get('href', '')
                                    if ('download' in link_text or 'pdf' in link_text) and href.endswith('.pdf'):
                                        pdf_link = urljoin(article_url, href)
                                        break
                                
                                if pdf_link:
                                    pdf_links.append({
                                        'url': pdf_link,
                                        'title': title,
                                        'abstract': abstract,
                                        'authors': authors,
                                        'article_url': article_url,
                                        'source': 'CORE.ac.uk',
                                        'language': 'en'
                                    })
                        except Exception as e:
                            print(f"Error fetching CORE article details: {e}")
            
            time.sleep(random.uniform(5, 7))
        except Exception as e:
            print(f"Error in CORE search ({query}): {e}")
    
    print(f"Found {len(pdf_links)} papers in CORE.ac.uk")
    return pdf_links

def search_sciencedirect_papers():
    """Search for breast cancer papers in ScienceDirect"""
    base_url = "https://www.sciencedirect.com/search"
    
    search_queries = [
        "breast cancer patient education",
        "breast cancer treatment options review",
        "breast cancer quality of life",
        "breast cancer communication patient",
        "breast cancer decision support"
    ]
    
    print("Searching for papers in ScienceDirect")
    
    pdf_links = []
    for query in search_queries:
        try:
            encoded_query = quote(query)
            # Filter for open access articles only, as these are more likely to be downloadable
            search_url = f"{base_url}?qs={encoded_query}&show=100&years=2019%2C2024&accessTypes=openaccess"
            
            response = requests.get(search_url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article results
                articles = soup.select(".result-item-content")
                
                for article in articles:
                    # Get title and link
                    title_elem = article.select_one(".result-list-title-link")
                    if not title_elem:
                        continue
                    
                    title = title_elem.text.strip()
                    article_url = title_elem.get('href', '')
                    if article_url and not article_url.startswith('http'):
                        article_url = "https://www.sciencedirect.com" + article_url
                    
                    # Get abstract snippet
                    abstract_elem = article.select_one(".result-item-content-text")
                    abstract = abstract_elem.text.strip() if abstract_elem else ""
                    
                    # Get journal info
                    journal_elem = article.select_one(".publication-title-link")
                    journal = journal_elem.text.strip() if journal_elem else ""
                    
                    if is_relevant_paper(article_url, title, abstract):
                        # For relevant articles, visit article page to find PDF
                        try:
                            article_response = requests.get(article_url, headers=HEADERS, timeout=15)
                            if article_response.status_code == 200:
                                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                                
                                # Try to find PDF link
                                pdf_link = None
                                for link in article_soup.select("a.pdf-download-link, a.download-pdf-link"):
                                    href = link.get('href', '')
                                    if href:
                                        if not href.startswith('http'):
                                            href = urljoin("https://www.sciencedirect.com", href)
                                        pdf_link = href
                                        break
                                
                                # Get DOI
                                doi = ""
                                doi_elem = article_soup.select_one(".doi")
                                if doi_elem:
                                    doi_text = doi_elem.text.strip()
                                    doi_match = re.search(r'10\.\d{4,}[\/\.].*$', doi_text)
                                    if doi_match:
                                        doi = doi_match.group(0)
                                
                                # Get full abstract
                                full_abstract = ""
                                abstract_section = article_soup.select_one(".abstract")
                                if abstract_section:
                                    full_abstract = abstract_section.text.strip()
                                
                                if pdf_link:
                                    pdf_links.append({
                                        'url': pdf_link,
                                        'title': title,
                                        'abstract': full_abstract or abstract,
                                        'journal': journal,
                                        'doi': doi,
                                        'article_url': article_url,
                                        'source': 'ScienceDirect',
                                        'language': 'en'
                                    })
                        except Exception as e:
                            print(f"Error fetching ScienceDirect article details: {e}")
            
            time.sleep(random.uniform(7, 10))
        except Exception as e:
            print(f"Error in ScienceDirect search ({query}): {e}")
    
    print(f"Found {len(pdf_links)} papers in ScienceDirect")
    return pdf_links

def search_guideline_organizations():
    """Search for breast cancer guidelines from guideline-producing organizations"""
    organizations = [
    {
        "name": "NCCN (National Comprehensive Cancer Network)",
        "base_url": "https://www.nccn.org",
        "search_path": "/patients/guidelines/cancers.aspx",
        "resource_type": "Patient Guidelines"
    },
    {
        "name": "ASCO (American Society of Clinical Oncology)",
        "base_url": "https://www.asco.org",
        "search_path": "/guidelines",
        "resource_type": "Clinical Guidelines"
    },
    {
        "name": "American Cancer Society",
        "base_url": "https://www.cancer.org",
        "search_path": "/cancer/breast-cancer.html",
        "resource_type": "Patient Information"
    },
    {
        "name": "Breastcancer.org",
        "base_url": "https://www.breastcancer.org",
        "search_path": "/treatment",
        "resource_type": "Patient Information"
    },
    {
        "name": "ESMO (European Society for Medical Oncology)",
        "base_url": "https://www.esmo.org",
        "search_path": "/guidelines/breast-cancer",
        "resource_type": "Clinical Guidelines"
    },
    {
        "name": "NICE (National Institute for Health and Care Excellence)",
        "base_url": "https://www.nice.org.uk",
        "search_path": "/guidance/conditions-and-diseases/cancer/breast-cancer",
        "resource_type": "Clinical Guidelines"
    },
    {
        "name": "Cancer.Net",
        "base_url": "https://www.cancer.net",
        "search_path": "/cancer-types/breast-cancer",
        "resource_type": "Patient Information"
    },
    {
        "name": "Susan G. Komen",
        "base_url": "https://www.komen.org",
        "search_path": "/breast-cancer/treatment",
        "resource_type": "Patient Information"
    },
    {
        "name": "CDC (Centers for Disease Control)",
        "base_url": "https://www.cdc.gov",
        "search_path": "/cancer/breast/",
        "resource_type": "Patient Information"
    },
    {
        "name": "WHO (World Health Organization)",
        "base_url": "https://www.who.int",
        "search_path": "/publications/i/item/9789241507936",
        "resource_type": "Clinical Guidelines"
    }
]
    
    print("Searching for guidelines from specialized organizations")
    
    pdf_links = []
    for org in organizations:
        try:
            search_url = org["base_url"] + org["search_path"]
            
            response = requests.get(search_url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for direct PDF links on the main page
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    link_text = link.get_text().strip()
                    
                    # Check if this link points to a PDF
                    if href.endswith('.pdf') or 'download' in href.lower() and 'pdf' in href.lower():
                        title = link_text or href.split('/')[-1].replace('.pdf', '')
                        full_url = href if href.startswith('http') else urljoin(org["base_url"], href)
                        
                        if is_relevant_guideline(full_url, title):
                            pdf_links.append({
                                'url': full_url,
                                'title': title,
                                'source': org["name"],
                                'language': 'en',
                                'resource_type': org["resource_type"]
                            })
                
                # Look for subpages that might contain PDFs
                # First, collect promising subpages
                subpages = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    link_text = link.get_text().strip().lower()
                    
                    # Look for links that might lead to guidelines/patient resources
                    if (('guideline' in link_text or 'guide' in link_text or 
                         'resource' in link_text or 'publication' in link_text or
                         'download' in link_text or 'pdf' in link_text or
                         'document' in link_text or 'information' in link_text) and
                        ('breast' in link_text or 'cancer' in link_text)):
                        
                        full_url = href if href.startswith('http') else urljoin(org["base_url"], href)
                        subpages.append(full_url)
                
                # Limit number of subpages to check to avoid too many requests
                for subpage_url in subpages[:5]:
                    try:
                        subpage_response = requests.get(subpage_url, headers=HEADERS, timeout=15)
                        if subpage_response.status_code == 200:
                            subpage_soup = BeautifulSoup(subpage_response.content, 'html.parser')
                            
                            # Look for PDF links on the subpage
                            for sub_link in subpage_soup.find_all('a', href=True):
                                sub_href = sub_link['href']
                                sub_text = sub_link.get_text().strip()
                                
                                # Check if this is a PDF link
                                if sub_href.endswith('.pdf') or ('download' in sub_href.lower() and 'pdf' in sub_href.lower()):
                                    title = sub_text or sub_href.split('/')[-1].replace('.pdf', '')
                                    full_sub_url = sub_href if sub_href.startswith('http') else urljoin(org["base_url"], sub_href)
                                    
                                    if is_relevant_guideline(full_sub_url, title):
                                        pdf_links.append({
                                            'url': full_sub_url,
                                            'title': title,
                                            'source': org["name"],
                                            'language': 'en',
                                            'resource_type': org["resource_type"],
                                            'found_at': subpage_url
                                        })
                            
                        time.sleep(random.uniform(2, 3))
                    except Exception as e:
                        print(f"Error processing subpage from {org['name']}: {e}")
            
            time.sleep(random.uniform(5, 8))
        except Exception as e:
            print(f"Error searching {org['name']}: {e}")
    
    print(f"Found {len(pdf_links)} guidelines from specialized organizations")
    return pdf_links

def download_pdfs(pdf_links, limit=100):
    """Download PDFs and save metadata"""
    downloaded = 0
    failed = 0
    metadata = []
    
    # Check for existing metadata file
    existing_metadata = []
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        except:
            print("Error reading existing metadata, starting fresh")
    
    # Get existing filenames and URLs to avoid duplicates
    existing_filenames = set()
    existing_urls = set()
    for item in existing_metadata:
        if 'filename' in item:
            existing_filenames.add(item['filename'])
        if 'url' in item:
            existing_urls.add(item['url'])
    
    # First, create a categorized list for better organization
    categorized_pdfs = {}
    
    for pdf in pdf_links:
        resource_type = pdf.get('resource_type', 'Other')
        if resource_type not in categorized_pdfs:
            categorized_pdfs[resource_type] = []
        categorized_pdfs[resource_type].append(pdf)
    
    # Process each category
    for resource_type, pdfs in categorized_pdfs.items():
        # Sort by priority if available
        pdfs.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # Process PDFs from this category
        for pdf in pdfs:
            if downloaded >= limit:
                break
                
            url = pdf['url']
            
            # Skip if already downloaded
            if url in existing_urls:
                print(f"Already downloaded: {pdf.get('title', url)}")
                continue
            
            title = pdf.get('title', 'Untitled')
            source = pdf.get('source', 'Unknown')
            
            # Create a better organized filename
            resource_prefix = resource_type.replace(' ', '_')
            source_prefix = source.replace(' ', '_')
            safe_title = format_filename(title)
            filename = f"{resource_prefix}_{source_prefix}_{safe_title}.pdf"
            
            # Ensure filename is unique
            if filename in existing_filenames:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{resource_prefix}_{source_prefix}_{safe_title}_{timestamp}.pdf"
            
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            # Skip if file already exists locally
            if os.path.exists(file_path):
                print(f"File already exists locally: {filename}")
                continue
            
            # Verify that the PDF is valid before attempting to download
            if not check_pdf_validity(url):
                print(f"URL does not point to a valid PDF: {url}")
                failed += 1
                continue
                
            try:
                print(f"Downloading: {title} ({resource_type})")
                response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
                
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    print(f"✓ Saved as: {filename}")
                    
                    # Prepare enhanced metadata
                    metadata_item = pdf.copy()
                    metadata_item['filename'] = filename
                    metadata_item['download_date'] = datetime.now().strftime("%Y-%m-%d")
                    metadata_item['file_path'] = file_path
                    
                    # Add RAG-specific metadata
                    metadata_item['rag_category'] = resource_type
                    
                    # Determine reading level based on resource type and audience
                    if pdf.get('audience') == 'Patients':
                        metadata_item['reading_level'] = 'General Public'
                    elif pdf.get('audience') == 'Healthcare Providers':
                        metadata_item['reading_level'] = 'Healthcare Professional'
                    else:
                        metadata_item['reading_level'] = 'Mixed'
                    
                    # Add content summary if not present
                    if 'summary' not in metadata_item and 'abstract' in metadata_item:
                        metadata_item['summary'] = metadata_item['abstract']
                    elif 'summary' not in metadata_item:
                        metadata_item['summary'] = f"A {resource_type.lower()} about breast cancer from {source}"
                    
                    # Filter sensitive or unnecessary fields
                    if 'full_text_links' in metadata_item:
                        metadata_item['has_full_text_links'] = bool(metadata_item['full_text_links'])
                        del metadata_item['full_text_links']
                    
                    metadata.append(metadata_item)
                    existing_urls.add(url)
                    existing_filenames.add(filename)
                    
                    downloaded += 1
                else:
                    print(f"✗ Error downloading PDF: {response.status_code}")
                    failed += 1
            
            except Exception as e:
                print(f"✗ Error during download: {e}")
                failed += 1
            
            # Pause to avoid overloading the server
            time.sleep(random.uniform(3, 5))
        
        if downloaded >= limit:
            break
    
    # Update metadata file
    all_metadata = existing_metadata + metadata
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    # Create a separate JSON with just the RAG-optimized metadata
    rag_metadata = []
    for item in all_metadata:
        rag_item = {
            'filename': item.get('filename', ''),
            'title': item.get('title', ''),
            'source': item.get('source', ''),
            'resource_type': item.get('resource_type', ''),
            'audience': item.get('audience', ''),
            'reading_level': item.get('reading_level', ''),
            'summary': item.get('summary', ''),
            'language': item.get('language', 'en'),
            'download_date': item.get('download_date', '')
        }
        rag_metadata.append(rag_item)
    
    rag_metadata_file = os.path.join(OUTPUT_DIR, 'rag_metadata.json')
    with open(rag_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(rag_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\nDownload Summary:")
    print(f"- Successfully downloaded {downloaded} guidelines/resources")
    print(f"- Failed downloads: {failed}")
    print(f"- Total resources in metadata: {len(all_metadata)}")
    print(f"- Created RAG-optimized metadata file: rag_metadata.json")
    
    return downloaded

def search_specific_guideline_urls():
    """Directly check known URLs that host breast cancer guidelines"""
    
    specific_urls = [
        # NCCN Guidelines
        {
            "url": "https://www.nccn.org/patients/guidelines/content/PDF/breast-invasive-patient.pdf",
            "title": "NCCN Guidelines for Patients: Breast Cancer - Invasive",
            "source": "NCCN",
            "resource_type": "Patient Guidelines"
        },
        {
            "url": "https://www.nccn.org/patients/guidelines/content/PDF/breast-noninvasive-patient.pdf",
            "title": "NCCN Guidelines for Patients: Breast Cancer - Noninvasive",
            "source": "NCCN",
            "resource_type": "Patient Guidelines"
        },
        {
            "url": "https://www.nccn.org/patients/guidelines/content/PDF/breast-metastatic-patient.pdf",
            "title": "NCCN Guidelines for Patients: Metastatic Breast Cancer",
            "source": "NCCN",
            "resource_type": "Patient Guidelines"
        },

        # ASCO Guidelines
        {
            "url": "https://ascopubs.org/doi/pdf/10.1200/JCO.20.03399",
            "title": "Systemic Therapy for Patients With Early-Stage HER2-Positive Breast Cancer",
            "source": "ASCO",
            "resource_type": "Clinical Guidelines"
        },

        # ESMO Guidelines
        {
            "url": "https://www.esmo.org/content/download/117593/2061518/1/ESMO-Patient-Guide-Survivorship.pdf",
            "title": "ESMO Patient Guide on Cancer Survivorship",
            "source": "ESMO",
            "resource_type": "Patient Guidelines"
        },

        # American Cancer Society Guidelines
        {
            "url": "https://www.cancer.org/cancer/types/breast-cancer/screening-tests-and-early-detection/american-cancer-society-recommendations-for-the-early-detection-of-breast-cancer.html",
            "title": "Breast Cancer - Early Detection",
            "source": "American Cancer Society",
            "resource_type": "Patient Information"
        },
        {
            "url": "https://www.cancer.org/cancer/types/breast-cancer.html",
            "title": "Breast Cancer Detailed Guide",
            "source": "American Cancer Society",
            "resource_type": "Patient Information"
        },

        # WHO Guidelines
        {
            "url": "https://iris.who.int/handle/10665/137339",
            "title": "WHO Position Paper on Mammography Screening",
            "source": "WHO",
            "resource_type": "Clinical Guidelines"
        },

        # Susan G. Komen
        {
            "url": "https://www.komen.org/support-resources/tools/questions-to-ask-your-doctor/",
            "title": "Questions to Ask Your Doctor About Breast Cancer",
            "source": "Susan G. Komen",
            "resource_type": "Patient Information"
        },

        # CDC Resources
        {
            "url": "https://www.cdc.gov/cancer/resources/print-materials.html",
            "title": "Breast Cancer Fact Sheet",
            "source": "CDC",
            "resource_type": "Patient Information"
        },

        # NICE Guidelines
        {
            "url": "https://www.nice.org.uk/guidance/ng101/resources/early-and-locally-advanced-breast-cancer-diagnosis-and-management-pdf-66141532913605",
            "title": "Early and Locally Advanced Breast Cancer: Diagnosis and Management",
            "source": "NICE",
            "resource_type": "Clinical Guidelines"
        },

        # Cancer Council Australia
        {
            "url": "https://www.cancer.org.au/cancer-information/downloadable-resources",
            "title": "Breast Cancer Treatment Options",
            "source": "Cancer Council Australia",
            "resource_type": "Patient Information"
        }
    ]
    
    print("Checking specific guideline URLs")
    
    valid_links = []
    for item in specific_urls:
        try:
            # Verify that the PDF is valid before adding
            if check_pdf_validity(item["url"]):
                valid_links.append(item)
                print(f"✓ Valid URL found: {item['title']}")
            else:
                print(f"✗ Invalid URL: {item['url']}")
        
        except Exception as e:
            print(f"Error checking URL {item['url']}: {e}")
        
        time.sleep(random.uniform(1, 2))
    
    print(f"Found {len(valid_links)} valid specific guideline URLs")
    return valid_links

def add_metadata_to_guidelines(guidelines):
    """Add additional useful metadata to guidelines"""
    for guideline in guidelines:
        # Add audience classification if not present
        if "resource_type" in guideline:
            resource_type = guideline["resource_type"].lower()
            
            # Determine the target audience
            if "patient" in resource_type or "information" in resource_type:
                guideline["audience"] = "Patients"
            elif "clinical" in resource_type or "practice" in resource_type:
                guideline["audience"] = "Healthcare Providers"
            else:
                guideline["audience"] = "Mixed/Unknown"
        else:
            # Try to determine from title
            title = guideline.get("title", "").lower()
            if "patient" in title or "information" in title or "booklet" in title:
                guideline["audience"] = "Patients"
                guideline["resource_type"] = "Patient Information"
            elif "guideline" in title or "clinical" in title or "consensus" in title:
                guideline["audience"] = "Healthcare Providers"
                guideline["resource_type"] = "Clinical Guidelines"
            else:
                guideline["audience"] = "Mixed/Unknown"
                guideline["resource_type"] = "Information Resource"
        
        # Add priority ranking based on source reputation and resource type
        # Higher priority for well-known guideline developers and patient-focused resources
        priority = 0
        
        # Source-based priority
        high_priority_sources = ["NCCN", "ASCO", "ESMO", "NICE", "WHO", "American Cancer Society"]
        medium_priority_sources = ["Breastcancer.org", "Susan G. Komen", "Cancer.Net", "CDC"]
        
        source = guideline.get("source", "")
        if any(org in source for org in high_priority_sources):
            priority += 3
        elif any(org in source for org in medium_priority_sources):
            priority += 2
        else:
            priority += 1
        
        # Type-based priority
        resource_type = guideline.get("resource_type", "")
        if "Guidelines" in resource_type:
            priority += 2
        elif "Patient Information" in resource_type:
            priority += 1
        
        guideline["priority"] = priority
    
    return guidelines

def main():
    print(f"=== BREAST CANCER GUIDELINES SCRAPER ===")
    print(f"This script will collect guidelines and information resources about breast cancer")
    print(f"with a focus on providing reliable information for patient counseling.\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect guidelines from all sources
    all_pdf_links = []
    
    # First, check specific known guideline URLs (highest reliability)
    specific_guidelines = search_specific_guideline_urls()
    all_pdf_links.extend(specific_guidelines)
    
    # Then search guideline-producing organizations
    org_guidelines = search_guideline_organizations()
    all_pdf_links.extend(org_guidelines)
    
    # Search PubMed for published guidelines
    pubmed_guidelines = search_pubmed_guidelines()
    all_pdf_links.extend(pubmed_guidelines)
    
    # Search Google Scholar
    google_scholar_guidelines = search_google_scholar_guidelines()
    all_pdf_links.extend(google_scholar_guidelines)
    
    # Remove duplicates based on URL
    unique_pdf_links = []
    seen_urls = set()
    
    for pdf in all_pdf_links:
        url = pdf['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique_pdf_links.append(pdf)
    
    # Add additional metadata
    unique_pdf_links = add_metadata_to_guidelines(unique_pdf_links)
    
    # Sort by priority (higher first)
    unique_pdf_links.sort(key=lambda x: x.get('priority', 0), reverse=True)
    
    print(f"\nTotal unique guidelines found: {len(unique_pdf_links)}")
    print(f"\nSources breakdown:")
    sources = {}
    for pdf in unique_pdf_links:
        source = pdf.get('source', 'Unknown')
        sources[source] = sources.get(source, 0) + 1
    
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"- {source}: {count}")
    
    print(f"\nResource type breakdown:")
    resource_types = {}
    for pdf in unique_pdf_links:
        resource_type = pdf.get('resource_type', 'Unknown')
        resource_types[resource_type] = resource_types.get(resource_type, 0) + 1
    
    for resource_type, count in sorted(resource_types.items(), key=lambda x: x[1], reverse=True):
        print(f"- {resource_type}: {count}")
    
    # Download PDFs (limited to 100 per execution)
    print("\nStarting downloads...")
    downloaded = download_pdfs(unique_pdf_links, limit=100)
    
    print(f"\nAll files saved to: {OUTPUT_DIR}")
    print(f"Guideline metadata saved to: {METADATA_FILE}")
    
    if downloaded == 0:
        print("\n⚠️ WARNING: No PDFs could be downloaded. Consider these possible issues:")
        print("1. Internet connection might be blocking requests to these sites")
        print("2. Some guidelines require registration or institutional access")
        print("3. You might need to use a VPN to access certain resources")
        print("4. Some sources may have changed their URLs or structure")
        print("5. You might be rate-limited by the websites")

def create_readme():
    """Create a README file explaining how to use the downloaded guidelines"""
    readme_content = f"""# Breast Cancer Guidelines Collection

## Overview
This folder contains guidelines and information resources about breast cancer, specifically collected 
to provide reliable information for patient counseling and a RAG (Retrieval-Augmented Generation) system.

## Content Description
The guidelines are organized by type:
- **Clinical Guidelines**: Professional medical guidelines from organizations like NCCN, ASCO, and ESMO
- **Patient Information**: Resources specifically designed for patients
- **Patient Guidelines**: Simplified guidelines adapted for patient understanding
- **Clinical Tools**: Decision aids, algorithms, and other tools for clinicians

## Metadata
Two JSON files are included:
1. `{os.path.basename(METADATA_FILE)}`: Complete metadata for all downloaded files
2. `rag_metadata.json`: Simplified metadata optimized for RAG systems

## Using with RAG Systems
To use these resources with a RAG system:
1. Index the PDFs using your preferred document processing pipeline
2. Use the `rag_metadata.json` to enhance your retrieval and generation
3. Consider categorizing responses based on the 'audience' and 'reading_level' fields

## Sources
Guidelines were collected from authoritative sources including:
- NCCN (National Comprehensive Cancer Network)
- ASCO (American Society of Clinical Oncology)
- ESMO (European Society for Medical Oncology)
- American Cancer Society
- National Institute for Health and Care Excellence (NICE)
- World Health Organization (WHO)
- And other reputable medical organizations

## Best Practices for RAG Implementation
When implementing a breast cancer counseling RAG system:
1. Prioritize recent guidelines (check the 'download_date' field)
2. Match the reading level to the user's needs
3. Clearly distinguish between patient information and clinical guidelines
4. Always include references to the source documents when providing information

Last updated: {datetime.now().strftime("%Y-%m-%d")}
"""
    
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Created README file: {readme_path}")
    return readme_path

if __name__ == "__main__":
    main()
    create_readme()