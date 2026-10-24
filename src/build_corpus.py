"""Descarga corpus en español para el RAG de HEAD-QA (FIR y BIR).

Fuentes (todas en español):
  wikipedia     Artículos de Wikipedia ES de categorías del temario FIR/BIR
                (farmacología, fisiología, bioquímica, inmunología...).
  libretexts    Libros de texto abiertos (CC) de espanol.libretexts.org: Biología
                (OpenStax), Anatomía y Fisiología (OpenStax), Microbiología
                (OpenStax), Principios de Farmacología, Bioquímica, etc.
  sefh          Capítulos del libro "Farmacia Hospitalaria" y el Manual de
                Farmacotecnia de la SEFH (PDF de su biblioteca virtual).
  pdfs_locales  Cualquier PDF que dejes en data/corpus/pdfs_locales/ (por ejemplo,
                manuales de academias FIR/BIR que tengas).
  cima          El corpus de fichas técnicas ya generado (data/corpus/cima.jsonl).

Ninguna fuente usa preguntas de HEAD-QA: el corpus es independiente del dataset.
Para medir la cobertura de los corpus descargados, ver src/coverage.py.

Uso:
  python src/build_corpus.py                          # todas salvo wikipedia
  python src/build_corpus.py --fuentes libretexts sefh

Salida:
  data/corpus/<fuente>.jsonl       pasajes {id, source, title, text}
  data/corpus/cache/<fuente>/      descargas en bruto (permite reanudar)
"""
import argparse
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import unquote

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CACHE_DIR = CORPUS_DIR / "cache"
USER_AGENT = "TAPLN-RAG-corpus/0.1 (proyecto universitario de PLN; python-requests)"

CHUNK_WORDS = 180    # tamaño de pasaje (palabras)
CHUNK_OVERLAP = 40   # solape entre pasajes consecutivos

# --- Wikipedia --------------------------------------------------------------
WIKI_API = "https://es.wikipedia.org/w/api.php"
WIKI_ESPERA = 1  # segundos entre peticiones (la API limita a los clientes anónimos)
WIKI_CATEGORIAS = [
    # Farmacia (FIR)
    "Farmacología", "Farmacocinética", "Farmacodinámica", "Toxicología",
    "Tecnología farmacéutica", "Formas farmacéuticas", "Antibióticos",
    "Analgésicos", "Antineoplásicos", "Antivirales", "Antifúngicos",
    "Inmunosupresores", "Diuréticos", "Hormonas", "Vitaminas",
    "Química farmacéutica", "Farmacognosia", "Química analítica", "Nutrición",
    # Biología (BIR) y ciencias básicas comunes
    "Fisiología humana", "Fisiología", "Bioquímica", "Metabolismo", "Enzimas",
    "Biología celular", "Biología molecular", "Genética", "Genética humana",
    "Inmunología", "Microbiología", "Bacteriología", "Virología", "Parasitología",
    "Micología", "Histología", "Neurociencia", "Endocrinología", "Hematología",
    "Embriología", "Proteínas", "Aminoácidos", "Lípidos", "Glúcidos",
    "Ácidos nucleicos", "Receptores celulares", "Neurotransmisores", "Bioestadística",
]
WIKI_PROFUNDIDAD = 2        # niveles de subcategorías
WIKI_MAX_POR_CATEGORIA = 800
# Subcategorías que suelen ser ruido (personas, historia, organizaciones...)
WIKI_EXCLUIR = re.compile(
    r"(ólogos|ólogas|científicos|investigador|médicos|farmacéuticos|premio|historia|"
    r"empresas|organizaciones|revistas|libros|películas|personas|por país|wikipedia|"
    r"plantillas|esbozos|marcas|deporte|dopaje|drogas recreativas|cultura)",
    re.IGNORECASE,
)

# --- LibreTexts (libros en español, licencias CC) ----------------------------
LIBRETEXTS = "https://espanol.libretexts.org/Bookshelves/"
LIBRETEXTS_LIBROS = [
    "Biologia/Biologia_introductoria_y_general/Libro%3A_Biologia_General_(OpenStax)",
    "Salud/Anatomia_y_Fisiologia/Libro%3A_Anatom%C3%ADa_y_Fisiolog%C3%ADa_1e_(OpenStax)",
    "Biologia/Microbiologia/Microbiolog%C3%ADa_(OpenStax)",
    "Salud/Farmacologia_y_Neurociencia/Libro%3A_Principios_de_Farmacologia_(Rosow_Standaert_y_Strichartz)",
    "Salud/Enfermeria/Farmacologia_de_Enfermeria_(OpenRN)",
    "Biologia/Bioquimica/Libro%3A_Bioqu%C3%ADmica_Libre_y_F%C3%A1cil_(Ahern_y_Rajagopal)",
    "Salud/Ciencias_Basicas/Biolog%C3%ADa_celular%2C_gen%C3%A9tica_y_bioqu%C3%ADmica_para_estudiantes_precl%C3%ADnicos",
    "Biologia/Biologia_Celular_y_Molecular/Libro%3A_Biologia_Celular_y_Molecular_Basica_(Bergtrom)",
    "Salud/Farmacologia_y_Neurociencia/Fundamentos_de_la_Neurociencia_(Henley)",
    "Salud/Anatomia_y_Fisiologia/Libro%3A_Fisiolog%C3%ADa_%C3%A1cido-base_(Brandis)",
    "Salud/Salud_Aliada/Libro%3A_Qu%C3%ADmica_Cl%C3%ADnica_-_Teor%C3%ADa%2C_An%C3%A1lisis%2C_Correlaci%C3%B3n_(Kaplan_y_Pesce)",
]
LIBRETEXTS_ESPERA = 5  # segundos entre peticiones (Crawl-delay de su robots.txt)
LIBRETEXTS_SALTAR = re.compile(r"(Front_Matter|Back_Matter|Materia_Frontal|Volver_Materia|/zz%3A|/00%3A)")

# --- SEFH (PDF de su biblioteca virtual) ------------------------------------
SEFH_PDFS = [
    *[f"https://www.sefh.es/bibliotecavirtual/fhtomo1/cap{n}.pdf"
      for n in (11, 12, 14, 15, 21, 22, 23, 24, 25, 28, 29)],
    "https://www.sefh.es/bibliotecavirtual/manual_farmacotecnia/manualFARMACOTECNIAFINAL_.pdf",
]

FUENTES = ["wikipedia", "libretexts", "sefh", "pdfs_locales", "cima"]
# Wikipedia no va por defecto: su API limita (429) a los clientes anónimos.
# Se puede pedir explícitamente con --fuentes wikipedia.
FUENTES_DEFECTO = [f for f in FUENTES if f != "wikipedia"]


# =============================================================================
# Utilidades
# =============================================================================
def sesion() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def get(s, url, reintentos=5, **kw):
    """GET con reintentos y espera exponencial ante 429 / 5xx."""
    for i in range(reintentos):
        try:
            r = s.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            espera = int(r.headers.get("Retry-After", 0)) or 2 ** (i + 1)
        except requests.ConnectionError:
            espera = 2 ** (i + 1)
        time.sleep(espera)
    raise RuntimeError(f"No se pudo descargar {url}")


def limpiar(texto: str) -> str:
    texto = texto.replace("​", "").replace("\xa0", " ")
    texto = re.sub(r"[ \t]+", " ", texto)
    return re.sub(r"\n\s*\n+", "\n", texto).strip()


def trocear(texto: str, n=CHUNK_WORDS, solape=CHUNK_OVERLAP):
    """Divide un texto en ventanas de n palabras con solape."""
    palabras = texto.split()
    if len(palabras) <= n:
        return [" ".join(palabras)] if palabras else []
    paso = n - solape
    return [" ".join(palabras[i:i + n]) for i in range(0, len(palabras) - solape, paso)]


def pasajes(fuente, doc_id, titulo, texto):
    for i, trozo in enumerate(trocear(limpiar(texto))):
        if len(trozo.split()) >= 20:  # descarta restos sin contenido
            yield {"id": f"{fuente}-{doc_id}-{i}", "source": fuente, "title": titulo, "text": trozo}


def guardar_jsonl(registros, ruta: Path):
    n = 0
    with open(ruta, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"  -> {ruta.relative_to(DATA_DIR.parent)}: {n} pasajes")


def leer_jsonl(ruta: Path):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clave(texto: str) -> str:
    return hashlib.md5(texto.encode("utf-8")).hexdigest()


# =============================================================================
# Descarga: Wikipedia
# =============================================================================
def wiki_get(s, params):
    time.sleep(WIKI_ESPERA)
    return get(s, WIKI_API, params=params).json()


def wiki_titulos_categoria(s, raiz):
    """Artículos de una categoría y sus subcategorías (BFS hasta WIKI_PROFUNDIDAD)."""
    titulos, vistas = [], {raiz}
    cola = deque([(f"Categoría:{raiz}", 0)])
    while cola and len(titulos) < WIKI_MAX_POR_CATEGORIA:
        cat, nivel = cola.popleft()
        params = {"action": "query", "list": "categorymembers", "cmtitle": cat,
                  "cmlimit": 500, "cmnamespace": "0|14", "format": "json"}
        while True:
            j = wiki_get(s, params)
            for m in j["query"]["categorymembers"]:
                nombre = m["title"].split(":", 1)[-1]
                if m["ns"] == 0:
                    titulos.append(m["title"])
                elif nivel < WIKI_PROFUNDIDAD and nombre not in vistas and not WIKI_EXCLUIR.search(nombre):
                    vistas.add(nombre)
                    cola.append((m["title"], nivel + 1))
            if "continue" not in j:
                break
            params.update(j["continue"])
    return titulos[:WIKI_MAX_POR_CATEGORIA]


def wiki_textos(s, titulos):
    """Texto plano de los artículos, de 50 en 50 (prop=cirrusdoc). Usa caché en disco."""
    cache = CACHE_DIR / "wikipedia" / "articulos.jsonl"
    textos = {}
    if cache.exists():
        textos = {r["title"]: r["text"] for r in leer_jsonl(cache)}
    pendientes = [t for t in dict.fromkeys(titulos) if t not in textos]
    print(f"    {len(textos)} artículos en caché, {len(pendientes)} por descargar")
    with open(cache, "a", encoding="utf-8") as f:
        for i in range(0, len(pendientes), 50):
            lote = pendientes[i:i + 50]
            j = wiki_get(s, {"action": "query", "prop": "cirrusdoc", "cdincludes": "text",
                                          "titles": "|".join(lote), "redirects": 1,
                                          "format": "json", "formatversion": 2})
            for p in j["query"].get("pages", []):
                docs = p.get("cirrusdoc") or []
                if docs and p["title"] not in textos:
                    textos[p["title"]] = docs[0]["source"].get("text", "")
                    f.write(json.dumps({"title": p["title"], "text": textos[p["title"]]}, ensure_ascii=False) + "\n")
            # Marca también los títulos que no existen / redirigen, para no repetirlos
            for t in lote:
                textos.setdefault(t, None)
            if (i // 50) % 20 == 0:
                print(f"    artículos: {i + len(lote)}/{len(pendientes)}")
    return {t: x for t, x in textos.items() if x}


def descargar_wikipedia():
    s = sesion()
    (CACHE_DIR / "wikipedia").mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "wikipedia" / "categorias.json"
    por_cat = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    titulos = []
    for cat in WIKI_CATEGORIAS:
        if cat not in por_cat:
            por_cat[cat] = wiki_titulos_categoria(s, cat)
            cache.write_text(json.dumps(por_cat, ensure_ascii=False), encoding="utf-8")
        print(f"    Categoría:{cat}: {len(por_cat[cat])} artículos")
        titulos += por_cat[cat]
    # Incluye todo lo descargado (también artículos a los que redirigía algún título)
    textos = wiki_textos(s, titulos)
    registros = (p for t, x in textos.items() for p in pasajes("wikipedia", clave(t)[:10], t, x))
    guardar_jsonl(registros, CORPUS_DIR / "wikipedia.jsonl")


# =============================================================================
# Descarga: LibreTexts
# =============================================================================
def libretexts_pagina(s, url):
    """Devuelve (título, texto, hijos) de una página; cachea cada página en disco."""
    from bs4 import BeautifulSoup

    ruta = CACHE_DIR / "libretexts" / f"{clave(url)}.json"
    if ruta.exists():
        d = json.loads(ruta.read_text(encoding="utf-8"))
        return d["title"], d["text"], d["children"]
    time.sleep(LIBRETEXTS_ESPERA)
    sopa = BeautifulSoup(get(s, url).text, "html.parser")
    url_base = url.rstrip("/")
    hijos = sorted({a["href"].split("#")[0] for a in sopa.select("a[href]")
                    if a["href"].startswith(url_base + "/") and not LIBRETEXTS_SALTAR.search(a["href"])})
    # Solo los hijos directos (un nivel más)
    hijos = [h for h in hijos if h[len(url_base) + 1:].count("/") == 0]
    h1 = sopa.select_one("h1")
    titulo = h1.get_text(" ", strip=True) if h1 else unquote(url_base.rsplit("/", 1)[-1])
    cuerpo = sopa.select_one("section.mt-content-container")
    texto = ""
    if cuerpo and not hijos:  # las páginas índice solo listan capítulos
        for basura in cuerpo.select("script, style, .mt-guide-content, .autoattribution, figure"):
            basura.decompose()
        texto = cuerpo.get_text("\n", strip=True)
    ruta.write_text(json.dumps({"url": url, "title": titulo, "text": texto, "children": hijos},
                               ensure_ascii=False), encoding="utf-8")
    return titulo, texto, hijos


def descargar_libretexts():
    s = sesion()
    (CACHE_DIR / "libretexts").mkdir(parents=True, exist_ok=True)
    registros = []
    for libro in LIBRETEXTS_LIBROS:
        raiz = LIBRETEXTS + libro
        nombre_libro, _, _ = libretexts_pagina(s, raiz)
        print(f"    {nombre_libro}")
        cola, vistas, n = deque([raiz]), {raiz}, 0
        while cola:
            url = cola.popleft()
            titulo, texto, hijos = libretexts_pagina(s, url)
            for h in hijos:
                if h not in vistas:
                    vistas.add(h)
                    cola.append(h)
            if texto:
                n += 1
                registros += pasajes("libretexts", clave(url)[:10], f"{nombre_libro}. {titulo}", texto)
        print(f"      {n} secciones con texto")
    guardar_jsonl(registros, CORPUS_DIR / "libretexts.jsonl")


# =============================================================================
# Descarga: PDF (SEFH y locales)
# =============================================================================
def texto_pdf(ruta: Path) -> str:
    from pypdf import PdfReader

    paginas = []
    for p in PdfReader(ruta).pages:
        t = p.extract_text() or ""
        t = re.sub(r"-\n(\w)", r"\1", t)  # une palabras cortadas con guion
        paginas.append(t)
    return "\n".join(paginas)


def descargar_sefh():
    s = sesion()
    carpeta = CACHE_DIR / "sefh"
    carpeta.mkdir(parents=True, exist_ok=True)
    registros = []
    for url in SEFH_PDFS:
        ruta = carpeta / url.split("bibliotecavirtual/")[-1].replace("/", "_")
        if not ruta.exists():
            try:
                ruta.write_bytes(get(s, url).content)
            except Exception as e:  # algún capítulo puede desaparecer de la web
                print(f"    [aviso] {url}: {e}")
                continue
        titulo = "SEFH. " + ruta.stem.replace("_", " ")
        registros += pasajes("sefh", ruta.stem, titulo, texto_pdf(ruta))
    guardar_jsonl(registros, CORPUS_DIR / "sefh.jsonl")


def descargar_pdfs_locales():
    carpeta = CORPUS_DIR / "pdfs_locales"
    carpeta.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(carpeta.glob("*.pdf"))
    if not pdfs:
        print(f"    No hay PDF en {carpeta.relative_to(DATA_DIR.parent)} (opcional)")
    registros = [p for ruta in pdfs for p in pasajes("pdfs_locales", clave(ruta.name)[:10],
                                                       ruta.stem, texto_pdf(ruta))]
    guardar_jsonl(registros, CORPUS_DIR / "pdfs_locales.jsonl")


DESCARGAS = {
    "wikipedia": descargar_wikipedia,
    "libretexts": descargar_libretexts,
    "sefh": descargar_sefh,
    "pdfs_locales": descargar_pdfs_locales,
    "cima": lambda: print("    Ya existe: se genera con tu script de CIMA"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fuentes", nargs="+", choices=FUENTES, default=FUENTES_DEFECTO)
    args = p.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for f in args.fuentes:
        print(f"[{f}]")
        DESCARGAS[f]()


if __name__ == "__main__":
    main()
