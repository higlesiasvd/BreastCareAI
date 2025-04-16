import requests
from bs4 import BeautifulSoup
import time
import os
import random
import re
from urllib.parse import urljoin, quote

# Configuración
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
    'Referer': 'https://www.google.com/'
}
OUTPUT_DIR = 'cancer_mama_pdfs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lista de palabras clave para filtrar PDFs relevantes
KEYWORDS_ES = ['cancer', 'cáncer', 'mama', 'pecho', 'mastectomía', 'oncología', 'tumor', 'quimioterapia', 'radioterapia', 'metástasis']
KEYWORDS_EN = ['breast', 'cancer', 'oncology', 'tumor', 'mammography', 'mastectomy', 'chemotherapy', 'radiation']

def is_relevant_pdf(url, title=""):
    """Comprobar si un PDF es relevante basado en palabras clave en la URL o título"""
    url_lower = url.lower()
    title_lower = title.lower()
    
    # Comprobar palabras clave en español
    for keyword in KEYWORDS_ES:
        if keyword in url_lower or keyword in title_lower:
            return True
            
    # Comprobar palabras clave en inglés
    for keyword in KEYWORDS_EN:
        if keyword in url_lower or keyword in title_lower:
            return True
            
    return False

def format_filename(s):
    """Convertir un string en un nombre de archivo seguro"""
    # Reemplazar caracteres no alfanuméricos con guiones bajos
    s = re.sub(r'[^\w\s-]', '_', s)
    # Reemplazar espacios con guiones
    s = re.sub(r'\s+', '-', s)
    # Limitar longitud del nombre
    return s[:100]

def check_pdf_validity(url):
    """Comprobar si una URL apunta a un PDF válido"""
    try:
        response = requests.head(url, headers=HEADERS, timeout=10)
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Verificar si es un PDF por el tipo de contenido
        if 'application/pdf' in content_type:
            # Verificar el tamaño para asegurarse de que no es muy pequeño (lo que podría indicar un error)
            content_length = int(response.headers.get('Content-Length', '0'))
            if content_length > 10000:  # Más de 10KB
                return True
        return False
    except Exception as e:
        print(f"Error verificando PDF {url}: {e}")
        return False

# Función para buscar PDFs en AECC (Asociación Española Contra el Cáncer)
def search_aecc_pdfs():
    """Buscar PDFs en la Asociación Española Contra el Cáncer"""
    base_url = "https://www.aecc.es"
    resource_urls = [
        f"{base_url}/es/todo-sobre-el-cancer/tipos-cancer/cancer-mama/",
        f"{base_url}/es/recursos/",
        f"{base_url}/es/publicaciones/",
    ]
    
    print("Buscando PDFs en AECC (Asociación Española Contra el Cáncer)")
    
    pdf_links = []
    for page_url in resource_urls:
        try:
            response = requests.get(page_url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Buscar enlaces a PDFs
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.endswith('.pdf'):
                        # Obtener título si existe
                        title = a_tag.get_text().strip() or href.split('/')[-1].replace('.pdf', '')
                        
                        # Asegurarse de que sea una URL completa
                        if not href.startswith('http'):
                            href = urljoin(base_url, href)
                        
                        if is_relevant_pdf(href, title):
                            pdf_links.append({
                                'url': href,
                                'title': title,
                                'source': 'AECC',
                                'language': 'es'
                            })
                
                # También buscar en páginas vinculadas
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if 'cancer-mama' in href or 'cancer/mama' in href:
                        subpage_url = urljoin(base_url, href)
                        try:
                            sub_response = requests.get(subpage_url, headers=HEADERS, timeout=15)
                            if sub_response.status_code == 200:
                                sub_soup = BeautifulSoup(sub_response.content, 'html.parser')
                                for sub_a_tag in sub_soup.find_all('a', href=True):
                                    sub_href = sub_a_tag['href']
                                    if sub_href.endswith('.pdf'):
                                        # Obtener título si existe
                                        sub_title = sub_a_tag.get_text().strip() or sub_href.split('/')[-1].replace('.pdf', '')
                                        
                                        # Asegurarse de que sea una URL completa
                                        if not sub_href.startswith('http'):
                                            sub_href = urljoin(base_url, sub_href)
                                        
                                        if is_relevant_pdf(sub_href, sub_title):
                                            pdf_links.append({
                                                'url': sub_href,
                                                'title': sub_title,
                                                'source': 'AECC',
                                                'language': 'es'
                                            })
                        except Exception as e:
                            print(f"Error al procesar subpágina de AECC: {e}")
            
            time.sleep(random.uniform(2, 4))
        except Exception as e:
            print(f"Error al procesar página de AECC: {e}")
    
    print(f"Se encontraron {len(pdf_links)} PDFs en AECC")
    return pdf_links

# Función para buscar PDFs en SEOM (Sociedad Española de Oncología Médica)
def search_seom_pdfs():
    """Buscar PDFs en la Sociedad Española de Oncología Médica"""
    base_url = "https://seom.org"
    resource_urls = [
        f"{base_url}/informacion-sobre-el-cancer/info-tipos-cancer/cancer-de-mama/",
        f"{base_url}/publicaciones/",
        f"{base_url}/recursos-para-pacientes/",
    ]
    
    print("Buscando PDFs en SEOM (Sociedad Española de Oncología Médica)")
    
    pdf_links = []
    for page_url in resource_urls:
        try:
            response = requests.get(page_url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Buscar enlaces a PDFs
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.endswith('.pdf'):
                        # Obtener título si existe
                        title = a_tag.get_text().strip() or href.split('/')[-1].replace('.pdf', '')
                        
                        # Asegurarse de que sea una URL completa
                        if not href.startswith('http'):
                            href = urljoin(base_url, href)
                        
                        if is_relevant_pdf(href, title):
                            pdf_links.append({
                                'url': href,
                                'title': title,
                                'source': 'SEOM',
                                'language': 'es'
                            })
            
            time.sleep(random.uniform(2, 4))
        except Exception as e:
            print(f"Error al procesar página de SEOM: {e}")
    
    print(f"Se encontraron {len(pdf_links)} PDFs en SEOM")
    return pdf_links

# Función para buscar PDFs en Ministerio de Salud de España
def search_mscbs_pdfs():
    """Buscar PDFs en el Ministerio de Sanidad de España"""
    base_url = "https://www.sanidad.gob.es"
    search_url = f"{base_url}/buscar/cancer+mama"
    
    print("Buscando PDFs en Ministerio de Sanidad")
    
    pdf_links = []
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Buscar enlaces a PDFs
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.endswith('.pdf'):
                    # Obtener título si existe
                    title = a_tag.get_text().strip() or href.split('/')[-1].replace('.pdf', '')
                    
                    # Asegurarse de que sea una URL completa
                    if not href.startswith('http'):
                        href = urljoin(base_url, href)
                    
                    if is_relevant_pdf(href, title):
                        pdf_links.append({
                            'url': href,
                            'title': title,
                            'source': 'Ministerio de Sanidad',
                            'language': 'es'
                        })
    except Exception as e:
        print(f"Error al procesar Ministerio de Sanidad: {e}")
    
    print(f"Se encontraron {len(pdf_links)} PDFs en Ministerio de Sanidad")
    return pdf_links

# Función para buscar PDFs en GEICAM (Grupo Español de Investigación en Cáncer de Mama)
def search_geicam_pdfs():
    """Buscar PDFs en GEICAM"""
    base_url = "https://www.geicam.org"
    resource_urls = [
        f"{base_url}/es/publicaciones/",
        f"{base_url}/es/pacientes/",
    ]
    
    print("Buscando PDFs en GEICAM")
    
    pdf_links = []
    for page_url in resource_urls:
        try:
            response = requests.get(page_url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Buscar enlaces a PDFs
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.endswith('.pdf'):
                        # Obtener título si existe
                        title = a_tag.get_text().strip() or href.split('/')[-1].replace('.pdf', '')
                        
                        # Asegurarse de que sea una URL completa
                        if not href.startswith('http'):
                            href = urljoin(base_url, href)
                        
                        if is_relevant_pdf(href, title):
                            pdf_links.append({
                                'url': href,
                                'title': title,
                                'source': 'GEICAM',
                                'language': 'es'
                            })
            
            time.sleep(random.uniform(2, 4))
        except Exception as e:
            print(f"Error al procesar página de GEICAM: {e}")
    
    print(f"Se encontraron {len(pdf_links)} PDFs en GEICAM")
    return pdf_links

# Función para buscar PDFs en hospitales españoles
def search_hospital_pdfs():
    """Buscar PDFs en hospitales españoles importantes"""
    hospitals = [
        {"name": "Hospital Clinic Barcelona", "url": "https://www.clinicbarcelona.org/asistencia/enfermedades/cancer-de-mama"},
        {"name": "Hospital La Paz Madrid", "url": "https://www.comunidad.madrid/hospital/lapaz/profesionales/investigacion-docencia"},
        {"name": "Hospital 12 de Octubre", "url": "https://www.comunidad.madrid/hospital/12octubre/profesionales/investigacion-docencia"}
    ]
    
    print("Buscando PDFs en hospitales españoles")
    
    pdf_links = []
    for hospital in hospitals:
        try:
            response = requests.get(hospital["url"], headers=HEADERS, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Buscar enlaces a PDFs
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.endswith('.pdf'):
                        # Obtener título si existe
                        title = a_tag.get_text().strip() or href.split('/')[-1].replace('.pdf', '')
                        
                        # Asegurarse de que sea una URL completa
                        if not href.startswith('http'):
                            href = urljoin(hospital["url"], href)
                        
                        if is_relevant_pdf(href, title):
                            pdf_links.append({
                                'url': href,
                                'title': title,
                                'source': hospital["name"],
                                'language': 'es'
                            })
            
            time.sleep(random.uniform(2, 4))
        except Exception as e:
            print(f"Error al procesar {hospital['name']}: {e}")
    
    print(f"Se encontraron {len(pdf_links)} PDFs en hospitales españoles")
    return pdf_links

# Función para buscar PDFs en Google (específico para cáncer de mama)
def search_google_pdfs(lang="es"):
    """Buscar PDFs en Google con búsquedas específicas"""
    base_url = "https://www.google.com/search"
    
    # Diferentes búsquedas según el idioma
    if lang == "es":
        queries = [
            "filetype:pdf cancer mama guía pacientes",
            "filetype:pdf cáncer mama tratamiento",
            "filetype:pdf cancer mama síntomas diagnóstico",
            "filetype:pdf cáncer mama información pacientes",
            "filetype:pdf cancer mama folleto"
        ]
    else:  # inglés
        queries = [
            "filetype:pdf breast cancer patient guide",
            "filetype:pdf breast cancer treatment",
            "filetype:pdf breast cancer symptoms diagnosis",
            "filetype:pdf breast cancer patient information",
            "filetype:pdf breast cancer brochure"
        ]
    
    print(f"Buscando PDFs en Google ({lang})...")
    
    pdf_links = []
    for query in queries:
        try:
            # Codificar la consulta para URL
            encoded_query = quote(query)
            search_url = f"{base_url}?q={encoded_query}&num=20"
            
            response = requests.get(search_url, headers=HEADERS, timeout=20)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extraer resultados
                for result in soup.select(".yuRUbf a"):
                    href = result.get('href', '')
                    if href.lower().endswith('.pdf'):
                        # Obtener título
                        title_elem = result.select_one("h3")
                        title = title_elem.text if title_elem else href.split('/')[-1].replace('.pdf', '')
                        
                        pdf_links.append({
                            'url': href,
                            'title': title,
                            'source': 'Google Search',
                            'language': lang
                        })
            
            # Importante esperar entre búsquedas para evitar bloqueos
            time.sleep(random.uniform(5, 10))
        except Exception as e:
            print(f"Error en búsqueda de Google ({query}): {e}")
    
    print(f"Se encontraron {len(pdf_links)} PDFs en Google ({lang})")
    return pdf_links

# Función para buscar PDFs en American Cancer Society (mejorada)
def search_acs_pdfs():
    """Buscar PDFs en American Cancer Society (versión mejorada)"""
    base_url = "https://www.cancer.org"
    resource_urls = [
        f"{base_url}/content/dam/cancer-org/cancer-control/en/booklets-flyers/",
        f"{base_url}/cancer/breast-cancer/about.html",
        f"{base_url}/cancer/breast-cancer/treatment.html",
        f"{base_url}/cancer/breast-cancer/screening-tests-and-early-detection.html",
        f"{base_url}/cancer/breast-cancer/risk-and-prevention.html"
    ]
    
    print("Buscando PDFs en American Cancer Society")
    
    pdf_links = []
    for page_url in resource_urls:
        try:
            response = requests.get(page_url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Buscar enlaces a PDFs (directos e indirectos)
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    
                    # Comprobar enlaces directos a PDFs
                    if href.endswith('.pdf'):
                        full_url = href if href.startswith('http') else urljoin(base_url, href)
                        title = a_tag.get_text().strip() or href.split('/')[-1].replace('.pdf', '')
                        
                        if is_relevant_pdf(full_url, title):
                            pdf_links.append({
                                'url': full_url,
                                'title': title,
                                'source': 'American Cancer Society',
                                'language': 'en'
                            })
                    
                    # Comprobar enlaces de descarga que podrían llevar a PDFs
                    elif 'download' in href.lower() or 'breast-cancer' in href.lower():
                        try:
                            sub_url = href if href.startswith('http') else urljoin(base_url, href)
                            sub_response = requests.get(sub_url, headers=HEADERS, timeout=10)
                            
                            if sub_response.status_code == 200:
                                sub_soup = BeautifulSoup(sub_response.content, 'html.parser')
                                
                                for sub_a in sub_soup.find_all('a', href=True):
                                    sub_href = sub_a['href']
                                    if sub_href.endswith('.pdf'):
                                        full_sub_url = sub_href if sub_href.startswith('http') else urljoin(base_url, sub_href)
                                        sub_title = sub_a.get_text().strip() or sub_href.split('/')[-1].replace('.pdf', '')
                                        
                                        if is_relevant_pdf(full_sub_url, sub_title):
                                            pdf_links.append({
                                                'url': full_sub_url,
                                                'title': sub_title,
                                                'source': 'American Cancer Society',
                                                'language': 'en'
                                            })
                        except Exception as sub_e:
                            pass  # Ignorar errores en subpáginas
            
            time.sleep(random.uniform(3, 5))
        except Exception as e:
            print(f"Error al procesar página de ACS: {e}")
    
    print(f"Se encontraron {len(pdf_links)} PDFs en American Cancer Society")
    return pdf_links

# Función para buscar PDFs en Susan G. Komen (mejorada)
def search_komen_pdfs():
    """Buscar PDFs en Susan G. Komen (versión mejorada)"""
    base_url = "https://www.komen.org"
    resource_urls = [
        f"{base_url}/support-resources/downloadable-educational-resources/",
        f"{base_url}/breast-cancer/",
        f"{base_url}/support-resources/"
    ]
    
    print("Buscando PDFs en Susan G. Komen")
    
    pdf_links = []
    for page_url in resource_urls:
        try:
            response = requests.get(page_url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Buscar enlaces directos a PDFs
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.endswith('.pdf'):
                        full_url = href if href.startswith('http') else urljoin(base_url, href)
                        title = a_tag.get_text().strip() or href.split('/')[-1].replace('.pdf', '')
                        
                        if is_relevant_pdf(full_url, title):
                            pdf_links.append({
                                'url': full_url,
                                'title': title,
                                'source': 'Susan G. Komen',
                                'language': 'en'
                            })
                
                # Buscar enlaces a páginas de recursos
                resource_links = []
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if 'resources' in href.lower() or 'breast-cancer' in href.lower() or 'download' in href.lower():
                        resource_links.append(href if href.startswith('http') else urljoin(base_url, href))
                
                # Visitar enlaces de recursos (limitado para evitar sobrecarga)
                for resource_link in resource_links[:5]:  # Limitar a 5 enlaces
                    try:
                        sub_response = requests.get(resource_link, headers=HEADERS, timeout=10)
                        if sub_response.status_code == 200:
                            sub_soup = BeautifulSoup(sub_response.content, 'html.parser')
                            
                            for sub_a in sub_soup.find_all('a', href=True):
                                sub_href = sub_a['href']
                                if sub_href.endswith('.pdf'):
                                    full_sub_url = sub_href if sub_href.startswith('http') else urljoin(base_url, sub_href)
                                    sub_title = sub_a.get_text().strip() or sub_href.split('/')[-1].replace('.pdf', '')
                                    
                                    if is_relevant_pdf(full_sub_url, sub_title):
                                        pdf_links.append({
                                            'url': full_sub_url,
                                            'title': sub_title,
                                            'source': 'Susan G. Komen',
                                            'language': 'en'
                                        })
                        
                        time.sleep(random.uniform(2, 3))
                    except Exception as sub_e:
                        pass  # Ignorar errores en subpáginas
            
            time.sleep(random.uniform(3, 5))
        except Exception as e:
            print(f"Error al procesar página de Komen: {e}")
    
    print(f"Se encontraron {len(pdf_links)} PDFs en Susan G. Komen")
    return pdf_links

# Función para descargar PDFs
def download_pdfs(pdf_links, limit=50):
    """Descargar PDFs encontrados"""
    downloaded = 0
    failed = 0
    
    # Ordenar PDFs: primero los de español, luego los de inglés
    sorted_pdfs = sorted(pdf_links, key=lambda x: 0 if x.get('language', 'en') == 'es' else 1)
    
    for pdf in sorted_pdfs:
        if downloaded >= limit:
            break
            
        url = pdf['url']
        title = pdf['title']
        source = pdf.get('source', 'Unknown')
        language = pdf.get('language', 'en')
        
        # Crear un nombre de archivo seguro
        safe_title = format_filename(f"{source}_{title}")
        filename = f"{safe_title}_{language}.pdf"
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # Evitar descargar si ya existe
        if os.path.exists(file_path):
            print(f"El archivo {filename} ya existe, omitiendo...")
            continue
        
        # Verificar que el PDF es válido antes de intentar descargarlo
        if not check_pdf_validity(url):
            print(f"La URL no apunta a un PDF válido: {url}")
            failed += 1
            continue
            
        try:
            print(f"Descargando: {title} ({language})")
            response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"✓ Guardado como: {filename}")
                downloaded += 1
            else:
                print(f"✗ Error al descargar PDF: {response.status_code}")
                failed += 1
        
        except Exception as e:
            print(f"✗ Error durante la descarga: {e}")
            failed += 1
        
        # Pausa para no sobrecargar el servidor
        time.sleep(random.uniform(2, 4))
    
    print(f"\nResumen de descargas:")
    print(f"- Se descargaron {downloaded} PDFs correctamente")
    print(f"- Fallaron {failed} descargas")
    
    return downloaded

# Función principal
def main():
    # Obtener PDFs de fuentes en español
    all_pdf_links = []
    
    # 1. Fuentes españolas
    print("\n=== FUENTES EN ESPAÑOL ===\n")
    aecc_pdfs = search_aecc_pdfs()
    all_pdf_links.extend(aecc_pdfs)
    
    seom_pdfs = search_seom_pdfs()
    all_pdf_links.extend(seom_pdfs)
    
    mscbs_pdfs = search_mscbs_pdfs()
    all_pdf_links.extend(mscbs_pdfs)
    
    geicam_pdfs = search_geicam_pdfs()
    all_pdf_links.extend(geicam_pdfs)
    
    hospital_pdfs = search_hospital_pdfs()
    all_pdf_links.extend(hospital_pdfs)
    
    # Búsqueda en Google (español)
    google_es_pdfs = search_google_pdfs(lang="es")
    all_pdf_links.extend(google_es_pdfs)
    
    # 2. Fuentes inglesas
    print("\n=== FUENTES EN INGLÉS ===\n")
    acs_pdfs = search_acs_pdfs()
    all_pdf_links.extend(acs_pdfs)
    
    komen_pdfs = search_komen_pdfs()
    all_pdf_links.extend(komen_pdfs)
    
    # Búsqueda en Google (inglés)
    google_en_pdfs = search_google_pdfs(lang="en")
    all_pdf_links.extend(google_en_pdfs)
    
    # Eliminar duplicados basados en la URL
    unique_pdf_links = []
    seen_urls = set()
    
    for pdf in all_pdf_links:
        url = pdf['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique_pdf_links.append(pdf)
    
    print(f"\nTotal de PDFs únicos encontrados: {len(unique_pdf_links)}")
    print(f"- En español: {sum(1 for pdf in unique_pdf_links if pdf.get('language') == 'es')}")
    print(f"- En inglés: {sum(1 for pdf in unique_pdf_links if pdf.get('language') == 'en')}")
    
    # Descargar PDFs (limitado a 50 por ejecución)
    print("\nIniciando descargas...")
    downloaded = download_pdfs(unique_pdf_links, limit=50)
    
    print(f"\nTodos los archivos se guardaron en la carpeta: {OUTPUT_DIR}")
    
    if downloaded == 0:
        print("\n⚠️ AVISO: No se pudieron descargar PDFs. Considera estos posibles problemas:")
        print("1. La conexión a internet podría estar bloqueando las solicitudes")
        print("2. Las fuentes pueden haber cambiado sus URLs o estructura")
        print("3. Podrías necesitar un VPN para acceder a algunos recursos")
        print("4. Algunas fuentes pueden requerir autenticación")

if __name__ == "__main__":
    main()