# Breast Cancer Multimodal Analysis and Support System

Un sistema integrado que combina visión por computador, aprendizaje automático avanzado, sistemas inteligentes y procesamiento del lenguaje natural para el análisis, soporte y educación sobre cáncer de mama.

## Descripción del Proyecto

Este proyecto implementa un sistema multimodal que integra cuatro componentes principales:

1. **Módulo de Análisis de Mamografías**: Procesamiento y análisis de imágenes mediante técnicas de visión por computador.
2. **Sistema de Modelado Predictivo**: Modelos de aprendizaje automático avanzado para mejorar la precisión diagnóstica.
3. **Módulo de Procesamiento de Lenguaje Natural**: Análisis de literatura médica y sistema de preguntas-respuestas.
4. **Marco de Integración**: Arquitectura multi-agente que coordina los diferentes componentes.

El enfoque principal está en el componente de Sistemas Inteligentes, que proporciona capacidades de razonamiento y coordina los diferentes componentes especializados.

## Estructura del Repositorio

```
/breast-cancer-analysis
  /agents                # Implementación de agentes
    /perception          # Agente de percepción para el análisis de imágenes
    /knowledge           # Agente de gestión del conocimiento
    /reasoning           # Agente de razonamiento e inferencia
    /nlp                 # Agente de procesamiento de lenguaje natural
    /interaction         # Agente de interacción con el usuario
  /knowledge-base        # Base de conocimiento y ontología
  /vision-module         # Módulo de visión por computador
  /nlp-module            # Módulo de procesamiento de lenguaje natural
  /ml-module             # Módulo de aprendizaje automático avanzado
  /integration           # Marco de integración y comunicación
  /utils                 # Utilidades comunes
  /data                  # Datos de ejemplo y preprocesados
  /notebooks             # Jupyter notebooks para experimentación
  /docs                  # Documentación
  /tests                 # Pruebas unitarias e integración
```

## Requisitos

- Python 3.8+
- Bibliotecas especificadas en `requirements.txt`

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/your-username/breast-cancer-analysis.git
cd breast-cancer-analysis

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Iniciar el sistema
python src/main.py
```

## Datasets

El sistema está diseñado para funcionar con los siguientes conjuntos de datos públicos:

- CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
- MIAS dataset

## Contribución

Este proyecto es parte de la asignatura de Sistemas Inteligentes (G24GXX3.32X) de la Universidad Intercontinental de la Empresa (UIE).

## Licencia

[MIT](LICENSE)
