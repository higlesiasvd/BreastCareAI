# download_papers.py
import os
import requests
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
import urllib.parse

def download_pubmed_papers(query, max_papers=3, output_dir="medical_papers"):
    """Descarga papers de PubMed Central basados en la consulta"""
    print(f"Descargando papers para: '{query}'")
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/"
    search_url = f"{base_url}articles/PMC/?term={urllib.parse.quote(query)}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Encontrar enlaces a papers
        paper_links = soup.find_all("a", class_="docsum-title")
        print(f"Encontrados {len(paper_links)} papers, descargando máximo {max_papers}")
        
        downloaded = 0
        for i, link in enumerate(paper_links):
            if downloaded >= max_papers:
                break
                
            paper_id = link['href'].split('/')[-1]
            paper_url = f"{base_url}articles/PMC{paper_id}/"
            
            try:
                paper_response = requests.get(paper_url)
                paper_soup = BeautifulSoup(paper_response.text, 'html.parser')
                
                # Buscar enlace PDF
                pdf_link = paper_soup.find("a", {"class": "int-view", "data-format": "pdf"})
                
                if pdf_link and 'href' in pdf_link.attrs:
                    pdf_url = "https://www.ncbi.nlm.nih.gov" + pdf_link['href']
                    
                    pdf_response = requests.get(pdf_url)
                    if pdf_response.status_code == 200:
                        filename = f"{output_dir}/pmc_{paper_id}.pdf"
                        
                        with open(filename, "wb") as f:
                            f.write(pdf_response.content)
                        
                        print(f"✓ Paper {downloaded+1}/{max_papers}: PMC{paper_id}")
                        downloaded += 1
                        time.sleep(2)  # Ser respetuoso con el servidor
            except Exception as e:
                print(f"Error descargando paper {paper_id}: {e}")
                continue
        
        return downloaded
    except Exception as e:
        print(f"Error en la búsqueda: {e}")
        return 0

def download_cancer_org_guidelines(output_dir="medical_papers"):
    """Descarga guías de la American Cancer Society"""
    print("Descargando guías oficiales sobre cáncer de mama...")
    urls = [
        "https://www.cancer.org/content/dam/CRC/PDF/Public/8577.00.pdf",  # Facts & Figures
        "https://www.cancer.org/content/dam/CRC/PDF/Public/8579.00.pdf",  # Early Detection
        "https://www.cancer.org/content/dam/CRC/PDF/Public/8580.00.pdf",  # Diagnosis
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    for i, url in enumerate(urls):
        try:
            filename = url.split('/')[-1]
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(f"{output_dir}/{filename}", "wb") as f:
                    f.write(response.content)
                print(f"✓ Guía {i+1}/{len(urls)}: {filename}")
                successful += 1
            else:
                print(f"✗ Error descargando {url}: Status code {response.status_code}")
            
            time.sleep(1)
        except Exception as e:
            print(f"✗ Error descargando {url}: {e}")
    
    return successful

if __name__ == "__main__":
    print("=== Descargando documentos sobre cáncer de mama ===")
    
    total = 0
    total += download_pubmed_papers("breast cancer treatment guidelines", 2)
    total += download_pubmed_papers("breast cancer diagnosis", 2)
    total += download_pubmed_papers("breast cancer screening", 2)
    total += download_cancer_org_guidelines()
    
    print(f"\n¡Descarga completada! Se descargaron {total} documentos.")
    
    # Listar documentos descargados
    print("\nDocumentos disponibles:")
    for i, doc in enumerate(os.listdir("medical_papers"), 1):
        print(f"{i}. {doc}")