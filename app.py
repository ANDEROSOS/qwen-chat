import sys
import io

# Configurar salida UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TextIteratorStreamer
from bs4 import BeautifulSoup
import base64
import requests
import torch
import numpy as np
from datetime import datetime
from collections import deque
import random
import re
import json
import time
import threading

app = Flask(__name__)

# Configuración
OCR_API_URL = "https://api-ocr-tesseract.onrender.com/ocr"

# User agents para búsqueda web
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
]

# Cargar modelos
print("[*] Cargando Qwen2.5-1.5B-Instruct...")
llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
llm_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
llm_model.eval()
print("[OK] Qwen2.5-1.5B cargado")

print("[*] Cargando modelo de embeddings...")
embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embed_model = AutoModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    torch_dtype=torch.float32
)
embed_model.eval()
print("[OK] Embeddings cargados")
print("[WEB] Sistema de busqueda web activado")

# ==================== SISTEMA DE BÚSQUEDA WEB ====================

def get_headers():
    """Retorna headers aleatorios para las peticiones"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
    }

def extract_page_content(url):
    """Extrae el contenido completo de una página web"""
    try:
        resp = requests.get(url, headers=get_headers(), timeout=10)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Eliminar scripts, styles, y elementos no deseados
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript', 'form']):
            tag.decompose()

        # Buscar contenido principal
        main_content = None
        for selector in ['article', 'main', '.content', '.post-content', '.entry-content', '#content', '.article-body', '.post-body']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find('body')

        if not main_content:
            return None

        # Extraer texto limpio
        text = main_content.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 20]
        text = '\n'.join(lines)

        # Limitar tamaño
        if len(text) > 3000:
            text = text[:3000] + "..."

        return text

    except Exception as e:
        print(f"Error extrayendo contenido de {url}: {e}")
        return None

def search_web_mojeek(query, max_results=3):
    """Busca en la web usando Mojeek y extrae contenido"""
    try:
        print(f"🔍 Buscando en web: {query}")

        resp = requests.get(
            "https://www.mojeek.com/search",
            params={"q": query, "fmt": "html"},
            headers=get_headers(),
            timeout=12
        )

        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []

        for item in soup.select('ul.results-standard > li')[:max_results]:
            try:
                link = item.find('a', class_='ob') or item.find('a')
                if link:
                    title = link.get_text(strip=True)
                    url = link.get('href', '')

                    desc = item.find('p', class_='s')
                    description = desc.get_text(strip=True) if desc else ""

                    if url and title:
                        # Extraer contenido completo de la página
                        full_content = extract_page_content(url)

                        results.append({
                            "title": title,
                            "url": url,
                            "description": description,
                            "content": full_content or description
                        })
            except Exception as e:
                print(f"Error procesando resultado: {e}")
                continue

        print(f"✅ Encontrados {len(results)} resultados")
        return results

    except Exception as e:
        print(f"Error en búsqueda web: {e}")
        return None

def search_duckduckgo(query, max_results=3):
    """Búsqueda alternativa con DuckDuckGo HTML"""
    try:
        print(f"🦆 Buscando en DuckDuckGo: {query}")

        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=get_headers(),
            timeout=12
        )

        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []

        for item in soup.select('.result')[:max_results]:
            try:
                link = item.select_one('.result__a')
                if link:
                    title = link.get_text(strip=True)
                    url = link.get('href', '')

                    # Limpiar URL de DuckDuckGo redirect
                    if 'uddg=' in url:
                        url = requests.utils.unquote(url.split('uddg=')[1].split('&')[0])

                    desc = item.select_one('.result__snippet')
                    description = desc.get_text(strip=True) if desc else ""

                    if url and title and url.startswith('http'):
                        full_content = extract_page_content(url)

                        results.append({
                            "title": title,
                            "url": url,
                            "description": description,
                            "content": full_content or description
                        })
            except:
                continue

        print(f"✅ DuckDuckGo: {len(results)} resultados")
        return results

    except Exception as e:
        print(f"Error en DuckDuckGo: {e}")
        return None

def search_brave(query, max_results=3):
    """Búsqueda alternativa con Brave Search"""
    try:
        print(f"🦁 Buscando en Brave: {query}")

        resp = requests.get(
            "https://search.brave.com/search",
            params={"q": query, "source": "web"},
            headers=get_headers(),
            timeout=12
        )

        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')
        results = []

        for item in soup.select('.snippet')[:max_results]:
            try:
                link = item.select_one('a.result-header')
                if link:
                    title = link.get_text(strip=True)
                    url = link.get('href', '')

                    desc = item.select_one('.snippet-description')
                    description = desc.get_text(strip=True) if desc else ""

                    if url and title and url.startswith('http'):
                        full_content = extract_page_content(url)

                        results.append({
                            "title": title,
                            "url": url,
                            "description": description,
                            "content": full_content or description
                        })
            except:
                continue

        print(f"✅ Brave: {len(results)} resultados")
        return results if results else None

    except Exception as e:
        print(f"Error en Brave: {e}")
        return None

def search_searxng(query, max_results=3):
    """Búsqueda usando instancias públicas de SearXNG (funciona en cloud)"""
    # Lista de instancias públicas de SearXNG
    searxng_instances = [
        "https://searx.be",
        "https://search.bus-hit.me",
        "https://searx.tiekoetter.com",
        "https://search.sapti.me",
        "https://searx.rasp.fr",
    ]
    
    for instance in searxng_instances:
        try:
            print(f"🔎 Buscando en SearXNG ({instance}): {query}")
            
            resp = requests.get(
                f"{instance}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "language": "es"
                },
                headers=get_headers(),
                timeout=15
            )
            
            if resp.status_code != 200:
                continue
            
            data = resp.json()
            results = []
            
            for item in data.get("results", [])[:max_results]:
                title = item.get("title", "")
                url = item.get("url", "")
                content = item.get("content", "")
                
                if url and title:
                    # Intentar obtener contenido completo
                    full_content = extract_page_content(url)
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "description": content,
                        "content": full_content or content
                    })
            
            if results:
                print(f"✅ SearXNG: {len(results)} resultados")
                return results
                
        except Exception as e:
            print(f"⚠️ Error con SearXNG {instance}: {e}")
            continue
    
    return None

def search_wikipedia(query, max_results=3):
    """Búsqueda usando Wikipedia API (funciona en cualquier servidor)"""
    try:
        print(f"📚 Buscando en Wikipedia: {query}")
        
        # Primero buscar artículos relacionados
        search_resp = requests.get(
            "https://es.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
                "utf8": 1
            },
            headers=get_headers(),
            timeout=15
        )
        
        if search_resp.status_code != 200:
            return None
        
        search_data = search_resp.json()
        search_results = search_data.get("query", {}).get("search", [])
        
        if not search_results:
            return None
        
        results = []
        
        for item in search_results:
            page_id = item.get("pageid")
            title = item.get("title", "")
            snippet = item.get("snippet", "").replace('<span class="searchmatch">', '').replace('</span>', '')
            
            # Obtener extracto completo del artículo
            extract_resp = requests.get(
                "https://es.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "pageids": page_id,
                    "prop": "extracts",
                    "exintro": False,
                    "explaintext": True,
                    "exsectionformat": "plain",
                    "format": "json"
                },
                headers=get_headers(),
                timeout=15
            )
            
            content = snippet
            if extract_resp.status_code == 200:
                extract_data = extract_resp.json()
                pages = extract_data.get("query", {}).get("pages", {})
                if str(page_id) in pages:
                    full_extract = pages[str(page_id)].get("extract", "")
                    if full_extract:
                        # Limitar a 3000 caracteres
                        content = full_extract[:3000] + "..." if len(full_extract) > 3000 else full_extract
            
            url = f"https://es.wikipedia.org/wiki/{title.replace(' ', '_')}"
            
            results.append({
                "title": f"Wikipedia: {title}",
                "url": url,
                "description": snippet,
                "content": content
            })
        
        print(f"✅ Wikipedia: {len(results)} resultados")
        return results if results else None
        
    except Exception as e:
        print(f"Error en Wikipedia: {e}")
        return None

def search_duckduckgo_instant(query, max_results=3):
    """Búsqueda usando DuckDuckGo Instant Answer API (más permisivo)"""
    try:
        print(f"🦆 Buscando en DuckDuckGo Instant API: {query}")
        
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            },
            headers=get_headers(),
            timeout=12
        )
        
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        results = []
        
        # Respuesta abstracta principal
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", query),
                "url": data.get("AbstractURL", ""),
                "description": data.get("Abstract", "")[:200],
                "content": data.get("Abstract", "")
            })
        
        # Temas relacionados
        for topic in data.get("RelatedTopics", [])[:max_results-1]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "url": topic.get("FirstURL", ""),
                    "description": topic.get("Text", "")[:200],
                    "content": topic.get("Text", "")
                })
        
        # Infobox si existe
        if data.get("Infobox") and data["Infobox"].get("content"):
            info_text = ""
            for item in data["Infobox"]["content"]:
                if item.get("label") and item.get("value"):
                    info_text += f"{item['label']}: {item['value']}\n"
            if info_text and results:
                results[0]["content"] += "\n\nInformación adicional:\n" + info_text
        
        print(f"✅ DuckDuckGo Instant: {len(results)} resultados")
        return results if results else None
        
    except Exception as e:
        print(f"Error en DuckDuckGo Instant: {e}")
        return None

def perform_web_search(query):
    """Realiza búsqueda web con múltiples fallbacks"""
    # Lista de motores de búsqueda a intentar (ordenados por compatibilidad con cloud)
    search_engines = [
        ("Wikipedia", search_wikipedia),          # Siempre funciona
        ("DuckDuckGo API", search_duckduckgo_instant),  # API oficial, más permisivo
        ("SearXNG", search_searxng),              # Metabuscador
        ("Mojeek", search_web_mojeek),
        ("DuckDuckGo HTML", search_duckduckgo),
        ("Brave", search_brave),
    ]
    
    for name, search_func in search_engines:
        try:
            print(f"🔍 Intentando con {name}...")
            results = search_func(query)
            if results and len(results) > 0:
                print(f"✅ Búsqueda exitosa con {name}")
                return results
        except Exception as e:
            print(f"⚠️ Error con {name}: {e}")
            continue
    
    print("❌ Todos los motores de búsqueda fallaron")
    return None

# ==================== SISTEMA DE SESIONES ====================

class SessionStore:
    def __init__(self, max_history=10):
        self.sessions = {}
        self.max_history = max_history

    def get_or_create(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'chunks': [],
                'full_text': '',
                'metadata': {},
                'conversation_history': deque(maxlen=self.max_history),
                'last_document': None,
                'web_search_enabled': True
            }
        return self.sessions[session_id]

    def add_message(self, session_id, role, content):
        session = self.get_or_create(session_id)
        session['conversation_history'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })

    def get_conversation_context(self, session_id, max_messages=4):
        session = self.get_or_create(session_id)
        history = list(session['conversation_history'])[-max_messages:]

        context = ""
        for msg in history:
            if msg['role'] == 'user':
                context += f"Usuario: {msg['content']}\n"
            else:
                context += f"Asistente: {msg['content']}\n"

        return context.strip()

    def clear_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

document_store = SessionStore(max_history=10)

# ==================== FUNCIONES DE EMBEDDINGS ====================

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    encoded_input = embed_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    with torch.no_grad():
        model_output = embed_model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings[0].cpu().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_chunks(query, chunks_with_embeddings, top_k=3):
    query_embedding = get_embedding(query)

    similarities = []
    for chunk, embedding in chunks_with_embeddings:
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# ==================== GENERACIÓN DE RESPUESTAS ====================

def generate_stream_with_web_context(question, web_results, conversation_history=""):
    """Genera respuesta con streaming usando información de la web"""
    
    # Construir contexto de búsqueda web
    web_context = ""
    sources = []
    
    if web_results:
        for i, result in enumerate(web_results, 1):
            web_context += f"\n--- Fuente {i}: {result['title']} ---\n"
            web_context += f"{result['content']}\n"
            sources.append(f"[{i}] {result['title']}: {result['url']}")
    
    system_msg = """Eres un asistente inteligente y veraz.
Tu OBJETIVO PRINCIPAL es responder usando EXCLUSIVAMENTE la información de búsqueda proporcionada.
NO inventes información. Si la respuesta no está en el contexto, di "No encontré esa información en los resultados de búsqueda".
Cita las fuentes numéricamente [1], [2], etc."""
    
    if conversation_history:
        prompt = f"""HISTORIAL DE CONVERSACIÓN:
{conversation_history}

INFORMACIÓN DE BÚSQUEDA WEB (USAR ESTA INFORMACIÓN COMO VERDAD ABSOLUTA):
{web_context}

PREGUNTA DEL USUARIO: {question}

Instrucciones:
1. Responde basándote SOLO en la Información de Búsqueda Web anterior.
2. Si la información no es suficiente, dilo honestamente.
3. NO uses tu conocimiento previo para contradecir o inventar datos fuera del contexto.
"""
    else:
        prompt = f"""INFORMACIÓN DE BÚSQUEDA WEB (USAR ESTA INFORMACIÓN COMO VERDAD ABSOLUTA):
{web_context}

PREGUNTA DEL USUARIO: {question}

Instrucciones:
1. Responde basándote SOLO en la Información de Búsqueda Web anterior.
2. Si la información no es suficiente, dilo honestamente.
3. NO uses tu conocimiento previo para contradecir o inventar datos fuera del contexto.
"""
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    
    text_input = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = llm_tokenizer([text_input], return_tensors="pt").to("cpu")
    
    # Configurar streamer
    streamer = TextIteratorStreamer(llm_tokenizer, skip_special_tokens=True, skip_prompt=True)
    
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    
    # Iniciar generación en thread separado
    thread = threading.Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens a medida que se generan
    for text in streamer:
        yield text
    
    thread.join()

def generate_stream_with_context(context, question, conversation_history=""):
    """Genera respuesta con streaming para análisis de documentos"""
    
    system_msg = """Eres un asistente experto en análisis de documentos.
Tu trabajo es responder preguntas basándote SOLO en el contenido del documento proporcionado.
Responde en español de forma clara, precisa y estructurada.
Si la pregunta no puede responderse con el documento, indícalo claramente."""
    
    if conversation_history:
        prompt = f"""HISTORIAL DE CONVERSACIÓN:
{conversation_history}

CONTENIDO DEL DOCUMENTO:
{context}

PREGUNTA DEL USUARIO: {question}

Responde basándote en el documento:"""
    else:
        prompt = f"""CONTENIDO DEL DOCUMENTO:
{context}

PREGUNTA DEL USUARIO: {question}

Responde basándote en el documento:"""
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    
    text_input = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = llm_tokenizer([text_input], return_tensors="pt").to("cpu")
    
    # Configurar streamer
    streamer = TextIteratorStreamer(llm_tokenizer, skip_special_tokens=True, skip_prompt=True)
    
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=800,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    
    # Iniciar generación en thread separado
    thread = threading.Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens a medida que se generan
    for text in streamer:
        yield text
    
    thread.join()

def generate_stream_free_chat(question, conversation_history=""):
    """Genera respuesta con streaming para chat libre"""
    
    system_msg = """Eres un asistente útil, amigable e inteligente.
Responde preguntas de forma clara, concisa y precisa en español.
Si no sabes algo, admítelo honestamente."""
    
    if conversation_history:
        prompt = f"""HISTORIAL DE CONVERSACIÓN:
{conversation_history}

PREGUNTA DEL USUARIO: {question}

Responde de forma natural:"""
    else:
        prompt = f"""PREGUNTA DEL USUARIO: {question}

Responde de forma natural:"""
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    
    text_input = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = llm_tokenizer([text_input], return_tensors="pt").to("cpu")
    
    # Configurar streamer
    streamer = TextIteratorStreamer(llm_tokenizer, skip_special_tokens=True, skip_prompt=True)
    
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=600,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    
    # Iniciar generación en thread separado
    thread = threading.Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens a medida que se generan
    for text in streamer:
        yield text
    
    thread.join()

# ==================== INTERFAZ HTML MODERNA ====================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen AI Chat</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0d0d0d;
            --bg-secondary: #171717;
            --bg-tertiary: #212121;
            --bg-hover: #2a2a2a;
            --border-color: #333333;
            --text-primary: #ececec;
            --text-secondary: #9a9a9a;
            --text-muted: #666666;
            --accent-color: #10a37f;
            --accent-hover: #0d8a6a;
            --user-bubble: #2f2f2f;
            --bot-bubble: transparent;
            --error-color: #ef4444;
            --warning-color: #f59e0b;
            --info-color: #3b82f6;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }

        .app-container {
            display: flex;
            height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            width: 260px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 12px;
        }

        .new-chat-btn {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 14px;
            background: transparent;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 20px;
        }

        .new-chat-btn:hover {
            background: var(--bg-hover);
        }

        .new-chat-btn svg {
            width: 18px;
            height: 18px;
        }

        .sidebar-section {
            margin-bottom: 24px;
        }

        .sidebar-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 8px 14px;
        }

        .sidebar-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 14px;
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .sidebar-item:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .sidebar-item.active {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        .sidebar-item svg {
            width: 18px;
            height: 18px;
            opacity: 0.7;
        }

        .sidebar-footer {
            margin-top: auto;
            padding-top: 12px;
            border-top: 1px solid var(--border-color);
        }

        .model-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }

        .model-badge .dot {
            width: 8px;
            height: 8px;
            background: var(--accent-color);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Main Chat Area */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-primary);
            position: relative;
        }

        .chat-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-primary);
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 15px;
            font-weight: 600;
        }

        .header-actions {
            display: flex;
            gap: 8px;
        }

        .header-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            background: transparent;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }

        .header-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        /* Toggle Switch */
        .toggle-container {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border-radius: 20px;
            font-size: 13px;
        }

        .toggle-label {
            color: var(--text-secondary);
        }

        .toggle-switch {
            position: relative;
            width: 40px;
            height: 22px;
            background: var(--bg-hover);
            border-radius: 11px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .toggle-switch.active {
            background: var(--accent-color);
        }

        .toggle-switch::after {
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 18px;
            height: 18px;
            background: white;
            border-radius: 50%;
            transition: all 0.3s;
        }

        .toggle-switch.active::after {
            left: 20px;
        }

        /* Messages Area */
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 0;
        }

        .messages-wrapper {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Welcome Screen */
        .welcome-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
            padding: 40px;
        }

        .welcome-logo {
            width: 64px;
            height: 64px;
            background: linear-gradient(135deg, var(--accent-color), #059669);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(16, 163, 127, 0.3);
        }

        .welcome-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 12px;
            background: linear-gradient(135deg, var(--text-primary), var(--text-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .welcome-subtitle {
            font-size: 16px;
            color: var(--text-secondary);
            max-width: 500px;
            line-height: 1.6;
            margin-bottom: 40px;
        }

        .quick-actions {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            max-width: 600px;
            width: 100%;
        }

        .quick-action {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: left;
        }

        .quick-action:hover {
            background: var(--bg-tertiary);
            border-color: var(--accent-color);
            transform: translateY(-2px);
        }

        .quick-action-icon {
            font-size: 20px;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }

        .quick-action-content h4 {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .quick-action-content p {
            font-size: 12px;
            color: var(--text-secondary);
        }

        /* Message Styles */
        .message {
            display: flex;
            gap: 16px;
            padding: 24px 0;
            border-bottom: 1px solid var(--border-color);
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message:last-child {
            border-bottom: none;
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
        }

        .message.bot .message-avatar {
            background: linear-gradient(135deg, var(--accent-color), #059669);
        }

        .message-content {
            flex: 1;
            min-width: 0;
        }

        .message-role {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        .message-text {
            font-size: 15px;
            line-height: 1.7;
            color: var(--text-primary);
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .message-text p {
            margin-bottom: 12px;
        }

        .message-text p:last-child {
            margin-bottom: 0;
        }

        /* Sources */
        .message-sources {
            margin-top: 16px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .sources-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .source-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 0;
            font-size: 13px;
            color: var(--info-color);
            text-decoration: none;
            transition: color 0.2s;
        }

        .source-item:hover {
            color: #60a5fa;
        }

        .source-item svg {
            width: 14px;
            height: 14px;
            opacity: 0.7;
        }

        /* Document Indicator */
        .message-document {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 12px;
        }

        .message-document svg {
            width: 16px;
            height: 16px;
            color: var(--warning-color);
        }

        /* Thinking Animation */
        .thinking {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px;
            color: var(--text-secondary);
            font-size: 14px;
        }

        .thinking-dots {
            display: flex;
            gap: 4px;
        }

        .thinking-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-color);
            border-radius: 50%;
            animation: thinking 1.4s infinite ease-in-out;
        }

        .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
        .thinking-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes thinking {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }

        /* Search Status */
        .search-status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 16px;
            background: rgba(16, 163, 127, 0.1);
            border: 1px solid rgba(16, 163, 127, 0.2);
            border-radius: 8px;
            font-size: 13px;
            color: var(--accent-color);
            margin-bottom: 12px;
        }

        .search-status svg {
            width: 18px;
            height: 18px;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        /* Input Area */
        .input-area {
            padding: 20px;
            background: var(--bg-primary);
            border-top: 1px solid var(--border-color);
        }

        .input-wrapper {
            max-width: 800px;
            margin: 0 auto;
        }

        .input-container {
            display: flex;
            align-items: flex-end;
            gap: 12px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            transition: all 0.2s;
        }

        .input-container:focus-within {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.1);
        }

        .input-actions {
            display: flex;
            gap: 4px;
        }

        .input-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            background: transparent;
            border: none;
            border-radius: 8px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }

        .input-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .input-btn svg {
            width: 20px;
            height: 20px;
        }

        #messageInput {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 15px;
            font-family: inherit;
            resize: none;
            max-height: 200px;
            line-height: 1.5;
        }

        #messageInput:focus {
            outline: none;
        }

        #messageInput::placeholder {
            color: var(--text-muted);
        }

        .send-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            background: var(--accent-color);
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            transition: all 0.2s;
        }

        .send-btn:hover:not(:disabled) {
            background: var(--accent-hover);
        }

        .send-btn:disabled {
            background: var(--bg-tertiary);
            color: var(--text-muted);
            cursor: not-allowed;
        }

        .send-btn svg {
            width: 18px;
            height: 18px;
        }

        /* File Preview */
        .file-preview {
            display: none;
            align-items: center;
            gap: 12px;
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 12px;
            margin-bottom: 12px;
        }

        .file-preview.show {
            display: flex;
        }

        .file-preview-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--warning-color), #d97706);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }

        .file-preview-info {
            flex: 1;
        }

        .file-preview-name {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 2px;
        }

        .file-preview-size {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .file-preview-close {
            width: 28px;
            height: 28px;
            background: var(--error-color);
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.2s;
        }

        .file-preview-close:hover {
            background: #dc2626;
            transform: scale(1.1);
        }

        .input-hint {
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 12px;
        }

        #fileInput {
            display: none;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }

            .quick-actions {
                grid-template-columns: 1fr;
            }

            .messages-wrapper {
                padding: 16px;
            }

            .message {
                gap: 12px;
                padding: 16px 0;
            }
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <div class="sidebar">
            <button class="new-chat-btn" onclick="clearConversation()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 5v14M5 12h14"/>
                </svg>
                Nueva conversación
            </button>

            <div class="sidebar-section">
                <div class="sidebar-title">Capacidades</div>
                <div class="sidebar-item active">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                    </svg>
                    Búsqueda Web
                </div>
                <div class="sidebar-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    Análisis de Documentos
                </div>
                <div class="sidebar-item">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                    Chat con Memoria
                </div>
            </div>

            <div class="sidebar-footer">
                <div class="model-badge">
                    <span class="dot"></span>
                    Qwen 2.5 - 1.5B
                </div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <div class="chat-header">
                <div class="header-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                        <path d="M2 17l10 5 10-5"/>
                        <path d="M2 12l10 5 10-5"/>
                    </svg>
                    Qwen AI Chat
                </div>
                <div class="header-actions">
                    <div class="toggle-container">
                        <span class="toggle-label">Búsqueda Web</span>
                        <div class="toggle-switch active" id="webSearchToggle" onclick="toggleWebSearch()"></div>
                    </div>
                </div>
            </div>

            <div class="messages-container" id="messagesContainer">
                <div class="messages-wrapper" id="messagesWrapper">
                    <div class="welcome-screen" id="welcomeScreen">
                        <div class="welcome-logo">🤖</div>
                        <h1 class="welcome-title">¿En qué puedo ayudarte?</h1>
                        <p class="welcome-subtitle">
                            Soy un asistente con acceso a internet. Puedo buscar información actualizada,
                            analizar documentos y mantener conversaciones con memoria contextual.
                        </p>
                        <div class="quick-actions">
                            <div class="quick-action" onclick="sendQuickMessage('¿Cuáles son las últimas noticias de tecnología?')">
                                <div class="quick-action-icon">🌐</div>
                                <div class="quick-action-content">
                                    <h4>Buscar en la web</h4>
                                    <p>Información actualizada de internet</p>
                                </div>
                            </div>
                            <div class="quick-action" onclick="document.getElementById('fileInput').click()">
                                <div class="quick-action-icon">📄</div>
                                <div class="quick-action-content">
                                    <h4>Analizar documento</h4>
                                    <p>Subir PDF o imagen</p>
                                </div>
                            </div>
                            <div class="quick-action" onclick="sendQuickMessage('Explícame qué es la inteligencia artificial')">
                                <div class="quick-action-icon">💡</div>
                                <div class="quick-action-content">
                                    <h4>Aprender algo nuevo</h4>
                                    <p>Explicaciones detalladas</p>
                                </div>
                            </div>
                            <div class="quick-action" onclick="sendQuickMessage('¿Qué puedes hacer?')">
                                <div class="quick-action-icon">❓</div>
                                <div class="quick-action-content">
                                    <h4>Ver capacidades</h4>
                                    <p>Conoce mis funciones</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-wrapper">
                    <div class="file-preview" id="filePreview">
                        <div class="file-preview-icon">📄</div>
                        <div class="file-preview-info">
                            <div class="file-preview-name" id="fileName">documento.pdf</div>
                            <div class="file-preview-size" id="fileSize">2.5 MB</div>
                        </div>
                        <button class="file-preview-close" onclick="clearFile()">×</button>
                    </div>
                    <div class="input-container">
                        <div class="input-actions">
                            <button class="input-btn" onclick="document.getElementById('fileInput').click()" title="Adjuntar archivo">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                                </svg>
                            </button>
                            <input type="file" id="fileInput" accept="image/*,application/pdf" onchange="handleFileSelect(event)">
                        </div>
                        <textarea
                            id="messageInput"
                            placeholder="Escribe un mensaje..."
                            rows="1"
                            onkeydown="handleKeyPress(event)"
                            oninput="autoResize(this)"
                        ></textarea>
                        <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M22 2L11 13"/>
                                <path d="M22 2l-7 20-4-9-9-4 20-7z"/>
                            </svg>
                        </button>
                    </div>
                    <div class="input-hint">
                        Qwen AI con búsqueda web integrada • Enter para enviar
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFile = null;
        let currentFilename = '';
        let hasDocument = false;
        let webSearchEnabled = true;
        const sessionId = 'session_' + Date.now();

        function toggleWebSearch() {
            webSearchEnabled = !webSearchEnabled;
            const toggle = document.getElementById('webSearchToggle');
            toggle.classList.toggle('active', webSearchEnabled);
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (!file) return;

            currentFilename = file.name;
            const reader = new FileReader();
            reader.onload = (e) => {
                currentFile = e.target.result;

                document.getElementById('fileName').textContent = file.name;
                document.getElementById('fileSize').textContent = (file.size / 1024 / 1024).toFixed(2) + ' MB';
                document.getElementById('filePreview').classList.add('show');
            };
            reader.readAsDataURL(file);
        }

        function clearFile() {
            currentFile = null;
            currentFilename = '';
            document.getElementById('filePreview').classList.remove('show');
            document.getElementById('fileInput').value = '';
        }

        function hideWelcome() {
            const welcome = document.getElementById('welcomeScreen');
            if (welcome) welcome.style.display = 'none';
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }

        function sendQuickMessage(message) {
            document.getElementById('messageInput').value = message;
            sendMessage();
        }

        async function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();

            if (!message && !currentFile) {
                return;
            }

            hideWelcome();

            const fileToSend = currentFile;
            const filenameToSend = currentFilename;

            addUserMessage(message || "Analiza este documento", fileToSend, filenameToSend);

            messageInput.value = '';
            messageInput.style.height = 'auto';

            showThinking();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        image: fileToSend,
                        filename: filenameToSend,
                        session_id: sessionId,
                        web_search: webSearchEnabled
                    })
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                // Iniciar lectura del stream
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                hideThinking();
                
                // Preparar mensaje del bot
                let botMessageData = addBotMessage("", [], false);
                let contentDiv = botMessageData.contentDiv;
                let accumulatedText = "";
                let isFirstChunk = true;
                let metadataProcessed = false;
                
                // Variable para guardar las fuentes y mostrarlas al final
                let pendingSources = [];
                
                let buffer = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk;
                    
                    // Procesar metadata en la primera línea
                    if (!metadataProcessed) {
                        const newlineIndex = buffer.indexOf('\\n');
                        if (newlineIndex !== -1) {
                            const metadataLine = buffer.substring(0, newlineIndex);
                            try {
                                const metadata = JSON.parse(metadataLine);
                                
                                // Guardar fuentes para mostrar al final
                                if (metadata.sources && metadata.sources.length > 0) {
                                    pendingSources = metadata.sources;
                                }
                                
                                // El badge de WEB sí lo mostramos inmediato para feedback visual
                                if (metadata.searched_web) {
                                    addWebSearchBadge(botMessageData.messageDiv);
                                }
                                
                                if (metadata.has_document) {
                                    hasDocument = true;
                                }
                                
                                if (metadata.error) {
                                    contentDiv.innerHTML = `<span style="color: var(--error-color)">${metadata.error}</span>`;
                                    return;
                                }
                                
                            } catch (e) {
                                console.error("Error parsing metadata:", e);
                            }
                            
                            buffer = buffer.substring(newlineIndex + 1);
                            metadataProcessed = true;
                        }
                    }
                    
                    // Procesar contenido del mensaje
                    if (metadataProcessed) {
                        accumulatedText += buffer;
                        buffer = "";
                        contentDiv.innerHTML = formatText(accumulatedText);
                        scrollToBottom();
                    }
                }
                
                // Al finalizar el stream, mostrar las fuentes si existen
                if (pendingSources.length > 0) {
                    updateBotMessageSources(botMessageData.messageDiv, pendingSources);
                    scrollToBottom();
                }
                
                // Añadir respuesta al historial localmente si es necesario, 
                // aunque el servidor ya lo hizo.

            } catch (error) {
                hideThinking();
                addBotMessage('Error de conexión: ' + error.message, [], false);
            }

            clearFile();
        }

        function addUserMessage(text, file, filename) {
            const wrapper = document.getElementById('messagesWrapper');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user';

            let fileHtml = '';
            if (file && filename) {
                fileHtml = `
                    <div class="message-document">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <polyline points="14 2 14 8 20 8"/>
                        </svg>
                        ${filename}
                    </div>
                `;
            }

            messageDiv.innerHTML = `
                <div class="message-avatar">👤</div>
                <div class="message-content">
                    <div class="message-role">Tú</div>
                    <div class="message-text">${escapeHtml(text)}</div>
                    ${fileHtml}
                </div>
            `;

            wrapper.appendChild(messageDiv);
            scrollToBottom();
        }

        function addBotMessage(text, sources = [], searchedWeb = false) {
            const wrapper = document.getElementById('messagesWrapper');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot';
            
            // Estructura base
            messageDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="message-role">Qwen AI <span class="web-badge-container"></span></div>
                    <div class="message-text">${formatText(text)}</div>
                    <div class="sources-container"></div>
                </div>
            `;

            wrapper.appendChild(messageDiv);
            scrollToBottom();
            
            // Retornar referencias para actualización dynamics
            return {
                messageDiv: messageDiv,
                contentDiv: messageDiv.querySelector('.message-text')
            };
        }
        
        function updateBotMessageSources(messageDiv, sources) {
            const container = messageDiv.querySelector('.sources-container');
            if (!sources || sources.length === 0) return;
            
            container.innerHTML = `
                <div class="message-sources">
                    <div class="sources-title">Fuentes consultadas</div>
                    ${sources.map(source => `
                        <a href="${source.split(': ')[1] || '#'}" target="_blank" class="source-item">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                                <polyline points="15 3 21 3 21 9"/>
                                <line x1="10" y1="14" x2="21" y2="3"/>
                            </svg>
                            ${source.split(': ')[0] || source}
                        </a>
                    `).join('')}
                </div>
            `;
        }
        
        function addWebSearchBadge(messageDiv) {
            const container = messageDiv.querySelector('.web-badge-container');
            container.innerHTML = `<span style="display: inline-flex; align-items: center; gap: 4px; font-size: 11px; color: var(--accent-color); margin-left: 8px;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                </svg>
                Web
            </span>`;
        }

        function showThinking() {
            const wrapper = document.getElementById('messagesWrapper');
            const thinkingDiv = document.createElement('div');
            thinkingDiv.id = 'thinkingIndicator';
            thinkingDiv.className = 'message bot';

            const searchText = webSearchEnabled ? 'Buscando en la web...' : 'Pensando...';

            thinkingDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="thinking">
                        <div class="thinking-dots">
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                        </div>
                        <span>${searchText}</span>
                    </div>
                </div>
            `;

            wrapper.appendChild(thinkingDiv);
            document.getElementById('sendBtn').disabled = true;
            scrollToBottom();
        }

        function hideThinking() {
            const thinking = document.getElementById('thinkingIndicator');
            if (thinking) thinking.remove();
            document.getElementById('sendBtn').disabled = false;
        }

        function scrollToBottom() {
            const container = document.getElementById('messagesContainer');
            setTimeout(() => {
                container.scrollTop = container.scrollHeight;
            }, 100);
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function formatText(text) {
            // Escapar HTML
            text = escapeHtml(text);

            // Convertir saltos de línea
            text = text.replace(/\\n/g, '<br>');

            // Negrita **texto**
            text = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');

            // Cursiva *texto*
            text = text.replace(/\\*(.*?)\\*/g, '<em>$1</em>');

            return text;
        }

        async function clearConversation() {
            try {
                await fetch('/clear_session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        session_id: sessionId
                    })
                });
            } catch (error) {
                console.error('Error:', error);
            }

            hasDocument = false;
            clearFile();

            const wrapper = document.getElementById('messagesWrapper');
            wrapper.innerHTML = `
                <div class="welcome-screen" id="welcomeScreen">
                    <div class="welcome-logo">🤖</div>
                    <h1 class="welcome-title">¿En qué puedo ayudarte?</h1>
                    <p class="welcome-subtitle">
                        Soy un asistente con acceso a internet. Puedo buscar información actualizada,
                        analizar documentos y mantener conversaciones con memoria contextual.
                    </p>
                    <div class="quick-actions">
                        <div class="quick-action" onclick="sendQuickMessage('¿Cuáles son las últimas noticias de tecnología?')">
                            <div class="quick-action-icon">🌐</div>
                            <div class="quick-action-content">
                                <h4>Buscar en la web</h4>
                                <p>Información actualizada de internet</p>
                            </div>
                        </div>
                        <div class="quick-action" onclick="document.getElementById('fileInput').click()">
                            <div class="quick-action-icon">📄</div>
                            <div class="quick-action-content">
                                <h4>Analizar documento</h4>
                                <p>Subir PDF o imagen</p>
                            </div>
                        </div>
                        <div class="quick-action" onclick="sendQuickMessage('Explícame qué es la inteligencia artificial')">
                            <div class="quick-action-icon">💡</div>
                            <div class="quick-action-content">
                                <h4>Aprender algo nuevo</h4>
                                <p>Explicaciones detalladas</p>
                            </div>
                        </div>
                        <div class="quick-action" onclick="sendQuickMessage('¿Qué puedes hacer?')">
                            <div class="quick-action-icon">❓</div>
                            <div class="quick-action-content">
                                <h4>Ver capacidades</h4>
                                <p>Conoce mis funciones</p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    </script>
</body>
</html>
"""

# ==================== RUTAS DEL SERVIDOR ====================

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "ocr_api": OCR_API_URL,
        "llm": "Qwen2.5-1.5B + paraphrase-multilingual-MiniLM-L12-v2",
        "web_search": "enabled"
    })

@app.route('/clear_session', methods=['POST'])
def clear_session():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        document_store.clear_session(session_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        image_data = data.get('image')
        filename = data.get('filename', 'documento.jpg')
        session_id = data.get('session_id', 'default')
        web_search_enabled = data.get('web_search', True)

        session = document_store.get_or_create(session_id)
        
        # Si hay imagen nueva, procesar OCR
        if image_data:
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]

            image_bytes = base64.b64decode(image_data)

            is_pdf = filename.lower().endswith('.pdf')
            mime_type = 'application/pdf' if is_pdf else 'image/jpeg'

            files = {'image': (filename, image_bytes, mime_type)}
            form_data = {'lang': 'spa', 'detailed': 'true'}

            print(f"[1/4] Extrayendo texto del documento...")

            try:
                ocr_response = requests.post(OCR_API_URL, files=files, data=form_data, timeout=120)
            except requests.exceptions.ConnectionError:
                return jsonify({
                    "success": False,
                    "error": "Error de conexión con el servidor OCR."
                }), 503

            if ocr_response.status_code != 200:
                return jsonify({
                    "success": False,
                    "error": "Error en API OCR."
                }), 500

            try:
                ocr_result = ocr_response.json()
            except:
                return jsonify({
                    "success": False,
                    "error": "Error al procesar OCR"
                }), 500

            if not ocr_result.get('success'):
                return jsonify({
                    "success": False,
                    "error": ocr_result.get('error', 'Error en OCR')
                }), 500

            extracted_text = ocr_result.get('text', '')
            metadata = ocr_result.get('metadata', {})

            if not extracted_text or len(extracted_text.strip()) < 10:
                return jsonify({
                    "success": True,
                    "response": "No pude extraer texto legible del documento. Intenta con una imagen más clara.",
                    "metadata": metadata,
                    "has_document": False,
                    "sources": [],
                    "searched_web": False
                })

            print(f"[2/4] Texto extraído: {len(extracted_text)} caracteres")

            # Crear chunks y embeddings
            print(f"[3/4] Creando embeddings...")
            chunks = chunk_text(extracted_text, chunk_size=400, overlap=50)

            chunks_with_embeddings = []
            for chunk in chunks:
                embedding = get_embedding(chunk)
                chunks_with_embeddings.append((chunk, embedding))

            # Guardar en sesión
            session['chunks'] = chunks_with_embeddings
            session['full_text'] = extracted_text
            session['metadata'] = metadata
            session['last_document'] = filename

            print(f"[4/4] Documento procesado")

        # Si no hay mensaje, dar resumen del documento
        if not message and session.get('chunks'):
            message = "Dame un resumen completo y estructurado del documento"

        if not message:
            return jsonify({
                "success": False,
                "error": "Por favor escribe un mensaje"
            }), 400

        # Agregar al historial
        document_store.add_message(session_id, 'user', message)

        print(f"[IA] Procesando: {message[:50]}...")

        # Obtener historial
        conversation_history = document_store.get_conversation_context(session_id, max_messages=4)

        def generate_response_stream():
            sources = []
            searched_web = False
            generator = None

            # DECIDIR ESTRATEGIA
            if session.get('chunks'):
                # CASO 1: RAG con documento
                relevant_chunks = find_relevant_chunks(message, session['chunks'], top_k=3)
                context = "\n\n".join([f"Sección {i+1}:\n{chunk}" for i, (chunk, sim) in enumerate(relevant_chunks)])
                generator = generate_stream_with_context(context, message, conversation_history)
            
            elif web_search_enabled:
                # CASO 2: Búsqueda Web
                print(f"[WEB] Iniciando busqueda web...")
                web_results = perform_web_search(message)
                searched_web = True
                
                if web_results:
                    for i, result in enumerate(web_results, 1):
                        sources.append(f"[{i}] {result['title']}: {result['url']}")
                    generator = generate_stream_with_web_context(message, web_results, conversation_history)
                else:
                    generator = generate_stream_free_chat(message, conversation_history)
                    yield json.dumps({
                        "sources": [],
                        "searched_web": True,
                        "has_document": False
                    }) + "\n"
                    yield "No pude encontrar información en la web, pero te respondo con lo que sé:\n\n"
                    # Nota: Aquí ya enviamos metadata, el loop de abajo continuará con el generador
            
            else:
                # CASO 3: Chat Libre
                generator = generate_stream_free_chat(message, conversation_history)

            # Enviar metadata primero (si no se envió en el fallback)
            if not (web_search_enabled and not sources and not session.get('chunks')):
                yield json.dumps({
                    "sources": sources,
                    "searched_web": searched_web,
                    "has_document": bool(session.get('chunks'))
                }) + "\n"

            # Stream tokens
            full_response = ""
            if generator:
                for text in generator:
                    full_response += text
                    yield text
            
            # Guardar en historial al finalizar
            document_store.add_message(session_id, 'assistant', full_response)
            print(f"[OK] Streaming finalizado")

        return Response(stream_with_context(generate_response_stream()), mimetype='text/plain')

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR: {error_trace}")
        return jsonify({
            "success": False,
            "error": f"Error interno: {str(e)}"
        }), 500

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR: {error_trace}")
        return jsonify({
            "success": False,
            "error": f"Error interno: {str(e)}"
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 7860))
    print(f"[*] Iniciando servidor en puerto {port}...")
    print(f"[WEB] Busqueda web: ACTIVADA")
    print(f"[DOC] Analisis de documentos: ACTIVADO")
    app.run(host='0.0.0.0', port=port, debug=False)
