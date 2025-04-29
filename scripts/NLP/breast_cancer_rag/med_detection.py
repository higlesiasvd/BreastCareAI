import streamlit as st
import re
import os

# Importar transformers con manejo de errores
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    transformers_available = True
except ImportError:
    transformers_available = False

class MedicalTermsDetector:
    """Detector de términos médicos relevantes usando modelos de transformers pre-entrenados"""
    
    def __init__(self):
        """Inicializa el detector cargando el modelo BioBERT para NER o usando un glosario"""
        # Cargar el glosario de términos médicos (esto siempre estará disponible)
        self.medical_terms_glossary = {
            # Procedimientos diagnósticos
            "mammogram": "An X-ray image of the breast used to detect early signs of breast cancer.",
            "biopsy": "A procedure to remove a small sample of tissue for laboratory testing to determine if cancer cells are present.",
            "ultrasound": "An imaging test that uses sound waves to create pictures of the inside of the breast.",
            "mri": "Magnetic Resonance Imaging - uses powerful magnets and radio waves to create detailed images of breast tissue.",
            "ct scan": "Computed Tomography scan - uses X-rays to create detailed images of the inside of the body.",
            "pet scan": "Positron Emission Tomography scan - a nuclear medicine imaging test that uses a radioactive tracer to visualize metabolic activity.",
            
            # Términos relacionados con el cáncer y su clasificación
            "carcinoma": "A type of cancer that starts in cells that make up the skin or the tissue lining organs.",
            "ductal carcinoma": "Cancer that begins in the milk ducts of the breast.",
            "lobular carcinoma": "Cancer that begins in the milk-producing glands (lobules) of the breast.",
            "metastasis": "The spread of cancer cells from the place where they first formed to another part of the body.",
            "stage": "A way to describe cancer based on how much cancer is in the body and where it has spread.",
            "grade": "A description of a tumor based on how abnormal the tumor cells and tissue look under a microscope.",
            "invasive": "Cancer that has spread beyond where it first developed into surrounding healthy tissues.",
            "noninvasive": "Cancer that has not spread beyond where it first developed (also called in situ).",
            "malignant": "Cancerous cells that can invade nearby tissue and spread to other parts of the body.",
            "benign": "Non-cancerous growths that do not spread to other parts of the body.",
            
            # Receptores y marcadores
            "her2": "Human Epidermal growth factor Receptor 2 - a protein that promotes cancer cell growth when overexpressed.",
            "estrogen receptor": "A protein found on breast cells that binds to the hormone estrogen. Some breast cancers need estrogen to grow.",
            "progesterone receptor": "A protein found on breast cells that binds to the hormone progesterone. Some breast cancers need progesterone to grow.",
            "triple negative": "Breast cancer that tests negative for estrogen receptors, progesterone receptors, and HER2 protein.",
            "hormone receptor positive": "Cancer cells that have receptors for either estrogen or progesterone, or both.",
            
            # Tratamientos
            "lumpectomy": "Surgery to remove a breast tumor and a small amount of normal tissue around it.",
            "mastectomy": "Surgery to remove all breast tissue from a breast as a way to treat or prevent breast cancer.",
            "radiation therapy": "Treatment that uses high doses of radiation to kill cancer cells and shrink tumors.",
            "chemotherapy": "Treatment that uses drugs to kill cancer cells or stop them from growing.",
            "hormone therapy": "Treatment that blocks or lowers the amount of hormones in the body to slow or stop the growth of cancer.",
            "targeted therapy": "Treatment that targets specific proteins or genes that contribute to cancer growth and survival.",
            "immunotherapy": "Treatment that helps the immune system fight cancer.",
            "neoadjuvant therapy": "Treatment given before the main treatment, usually to shrink a tumor before surgery.",
            "adjuvant therapy": "Treatment given after the main treatment to lower the risk of cancer coming back.",
            
            # Efectos secundarios y complicaciones
            "lymphedema": "Swelling caused by a build-up of lymph fluid, often in an arm or leg after lymph node removal or damage.",
            "neutropenia": "An abnormally low count of neutrophils (a type of white blood cell), increasing the risk of infection.",
            "neuropathy": "Nerve damage that can cause numbness, tingling, or pain in the hands and feet.",
            "fatigue": "Extreme tiredness that doesn't get better with rest, a common side effect of cancer treatment.",
            
            # Genética
            "brca1": "A gene that, when mutated, increases the risk of breast and ovarian cancer.",
            "brca2": "A gene that, when mutated, increases the risk of breast and ovarian cancer.",
            "genetic testing": "Tests done to look for gene mutations associated with a higher risk of developing certain cancers.",
            
            # Supervivencia y seguimiento
            "recurrence": "The return of cancer after a period when no cancer could be detected.",
            "remission": "A decrease in or disappearance of signs and symptoms of cancer.",
            "survival rate": "The percentage of people who are alive after a certain period of time following a cancer diagnosis.",
            "follow-up care": "Regular medical check-ups after cancer treatment to monitor recovery and check for recurrence."
        }
    
        # Intentar cargar BioBERT solo si transformers está disponible
        if transformers_available:
            self.ner_pipeline = self._load_biobert_model()
        else:
            st.warning("Transformers library not available. Using dictionary-based detection only.")
            self.ner_pipeline = None

    @st.cache_resource
    def _load_biobert_model(_self):
        """Carga el modelo BioBERT para detección de términos médicos"""
        try:
            st.info("Loading biomedical NLP model (this may take a moment)...")
            
            model_name = "dmis-lab/biobert-base-cased-v1.1"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            return pipeline("ner", model=model, tokenizer=tokenizer)
        except Exception as e:
            st.warning(f"Could not load biomedical NLP model: {e}. Using dictionary-based detection only.")
            return None
                
    
    def detect_medical_terms(self, text):
        """Detecta términos médicos relevantes usando el glosario y BioBERT si está disponible"""
        if not text:
            return []
        
        results = []
        text_lower = text.lower()
        
        # Método 1: Buscar en el glosario (esto siempre funcionará)
        for term, definition in self.medical_terms_glossary.items():
            if term.lower() in text_lower:
                results.append({
                    "term": term,
                    "definition": definition,
                    "context": self._get_context(text, term)
                })
        
        # Método 2: Usar BioBERT solo si está disponible
        if self.ner_pipeline:
            try:
                # Procesar solo los primeros 512 tokens (límite típico de BERT)
                ner_results = self.ner_pipeline(text[:5000])
                for entity in ner_results:
                    if entity.get('entity', '').startswith('B-'):  # Entidades biomédicas
                        term = text[entity['start']:entity['end']]
                        # Evitar duplicados
                        if not any(r['term'].lower() == term.lower() for r in results):
                            results.append({
                                "term": term,
                                "definition": "Término biomédico identificado por BioBERT",
                                "context": self._get_context(text, term)
                            })
            except Exception as e:
                st.error(f"Error en BioBERT: {e}")
                # Continuar solo con los resultados del glosario
        
        return results
    
    def _get_context(self, text, term):
        """Extrae el contexto alrededor del término"""
        term_lower = term.lower()
        text_lower = text.lower()
        
        index = text_lower.find(term_lower)
        if index == -1:
            return ""
        
        start = max(0, index - 50)
        end = min(len(text), index + len(term) + 50)
        
        context = text[start:end]
        context = context.replace(term, f"**{term}**")
        
        return "..." + context + "..."
    
    def format_results_for_display(self, terms):
        """Formatea los resultados para mostrarlos en la interfaz de usuario"""
        if not terms:
            return "No relevant medical terms detected."
        
        # Solución: usar el formato directo sin categorías
        html_output = []
        for term in terms:
            term_name = term['term'].capitalize()
            definition = term.get('definition', "No detailed explanation available.")
            context = term.get('context', "")
            
            # Crear una tarjeta estilizada para cada término
            html_output.append(f"""
            <div style='background-color: #f8f9fa; border-left: 4px solid #4CAF50; 
                        margin-bottom: 10px; padding: 10px; border-radius: 4px;'>
                <span style='color: #2E7D32; font-weight: bold;'>{term_name}</span>
                <p style='margin-top: 5px; font-size: 0.9em;'>{definition}</p>
                <p style='margin-top: 5px; font-size: 0.85em; font-style: italic;'>{context}</p>
            </div>
            """)
        
        return "".join(html_output)
    
    # Función para obtener el detector
    def get_medical_terms_detector():
        """Retorna una instancia del detector de términos médicos"""
        return MedicalTermsDetector()

    # Detector simple como respaldo garantizado
    def get_basic_medical_terms_detector():
        """Versión simplificada que siempre funciona"""
        return MedicalTermsDetector()  # Usa el mismo detector
    
    def _categorize_term(self, term, entity_type=None):
        """Categoriza un término médico basándose en su contenido"""
        diagnostic_keywords = ["mammogram", "biopsy", "ultrasound", "mri", "ct", "pet", "scan", "imaging", "test"]
        cancer_keywords = ["carcinoma", "cancer", "tumor", "malignant", "benign", "stage", "grade", "invasive", "metastasis"]
        treatment_keywords = ["therapy", "treatment", "surgery", "radiation", "chemo", "lumpectomy", "mastectomy", "hormone"]
        receptor_keywords = ["receptor", "her2", "estrogen", "progesterone", "triple"]
        genetic_keywords = ["gene", "genetic", "brca", "dna", "mutation"]
        side_effect_keywords = ["effect", "lymphedema", "neuropathy", "fatigue", "pain", "nausea"]
        
        term_lower = term.lower()
        
        # Primero revisar keywords específicos
        if any(kw in term_lower for kw in diagnostic_keywords):
            return "Diagnostic Procedures"
        elif any(kw in term_lower for kw in cancer_keywords):
            return "Cancer Types & Classification"
        elif any(kw in term_lower for kw in treatment_keywords):
            return "Treatments"
        elif any(kw in term_lower for kw in receptor_keywords):
            return "Receptors & Markers"
        elif any(kw in term_lower for kw in genetic_keywords):
            return "Genetics"
        elif any(kw in term_lower for kw in side_effect_keywords):
            return "Side Effects"
        
        # Si no se encontró una categoría específica, usar información del modelo NER
        if entity_type:
            if "DISEASE" in entity_type:
                return "Conditions"
            elif "TREATMENT" in entity_type or "PROCEDURE" in entity_type:
                return "Treatments"
            elif "CHEMICAL" in entity_type:
                return "Medications"
        
        # Categoría por defecto
        return "Other Medical Terms"
    
    def _split_text(self, text, max_length):
        """Divide el texto en segmentos de tamaño máximo max_length"""
        words = text.split()
        segments = []
        current_segment = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length:
                segments.append(' '.join(current_segment))
                current_segment = [word]
                current_length = len(word)
            else:
                current_segment.append(word)
                current_length += len(word) + 1  # +1 para el espacio
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments
    
    def format_results_for_display(self, terms):
        """Formatea los resultados para mostrarlos en la interfaz de usuario"""
        if not terms:
            return "No relevant medical terms detected."
        
        # Solución: usar el formato directo sin categorías
        html_output = []
        for term in terms:
            term_name = term['term'].capitalize()
            definition = term.get('definition', "No detailed explanation available.")
            context = term.get('context', "")
            
            # Crear una tarjeta estilizada para cada término
            html_output.append(f"""
            <div style='background-color: #f8f9fa; border-left: 4px solid #4CAF50; 
                        margin-bottom: 10px; padding: 10px; border-radius: 4px;'>
                <span style='color: #2E7D32; font-weight: bold;'>{term_name}</span>
                <p style='margin-top: 5px; font-size: 0.9em;'>{definition}</p>
                <p style='margin-top: 5px; font-size: 0.85em; font-style: italic;'>{context}</p>
            </div>
            """)
        
        return "".join(html_output)


# Función para obtener o crear la instancia única del detector
def get_medical_terms_detector():
    """Obtiene o crea una instancia única del detector de términos médicos"""
    if 'medical_terms_detector' not in st.session_state:
        st.session_state.medical_terms_detector = MedicalTermsDetector()
    return st.session_state.medical_terms_detector