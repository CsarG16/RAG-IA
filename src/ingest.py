import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Asegurar path de src
sys.path.append(str(Path(__file__).resolve().parent))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNKS_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_table_as_markdown(table) -> str:
    """Convierte una tabla de PPTX en formato Markdown tabular."""
    rows_data = []
    for row in table.rows:
        row_text = [
            " ".join(cell.text.split()) if hasattr(cell, "text") else ""
            for cell in row.cells
        ]
        row_cleaned = [c.replace("|", "/") if c else "-" for c in row_text]
        rows_data.append(row_cleaned)

    if not rows_data or len(rows_data) < 1:
        return ""

    headers = rows_data[0]
    separator = ["---"] * len(headers)
    body = rows_data[1:]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |"
    ]
    for row in body:
        if len(row) == len(headers):
            lines.append("| " + " | ".join(row) + " |")

    return "\n" + "\n".join(lines) + "\n"


def extract_from_shape(shape, elements_list: List[str]):
    """Extrae texto y tablas de una forma, manejando recursivamente grupos (GroupShape)."""
    if hasattr(shape, "shapes"):
        for sub_shape in shape.shapes:
            extract_from_shape(sub_shape, elements_list)
        return

    if shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            text = paragraph.text.strip()
            if text:
                elements_list.append(text)
    elif shape.has_table:
        table_md = extract_table_as_markdown(shape.table)
        if table_md.strip():
            elements_list.append(f"\n[Tabla de Transición]:\n{table_md}\n")


def extract_text_from_pptx(file_path: Path) -> List[Dict[str, Any]]:
    """Extrae texto, tablas estructuradas y notas diapositiva por diapositiva desde un PPTX."""
    try:
        from pptx import Presentation
    except ImportError:
        logger.error("python-pptx no está instalado.")
        return []

    prs = Presentation(file_path)
    chunks = []
    file_name = file_path.name
    topic = file_path.stem.replace("_", " ").title()

    for idx, slide in enumerate(prs.slides, start=1):
        slide_elements = []
        for shape in slide.shapes:
            extract_from_shape(shape, slide_elements)

        # Extraer notas del orador si existen
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes and len(notes) > 5 and not notes.lower().startswith("diapositiva"):
                slide_elements.append(f"\n[Notas de Cátedra]: {notes}")

        # Evitar duplicados idénticos consecutivos
        dedup_elements = []
        for elem in slide_elements:
            elem_clean = elem.strip()
            if elem_clean and (not dedup_elements or dedup_elements[-1] != elem_clean):
                dedup_elements.append(elem_clean)

        slide_content = "\n".join(dedup_elements).strip()
        if slide_content:
            chunks.append({
                "id": f"{file_path.stem}_slide_{idx}",
                "source": file_name,
                "file_type": "pptx",
                "page_or_slide": idx,
                "topic": topic,
                "text": slide_content
            })

    logger.info(f"Extraídas {len(chunks)} diapositivas con texto y tablas de {file_name}")
    return chunks


def extract_text_from_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """Extrae texto página por página desde un archivo PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf no está instalado.")
        return []

    reader = PdfReader(file_path)
    chunks = []
    file_name = file_path.name
    topic = file_path.stem.replace("_", " ").title()

    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        clean_text = page_text.strip()
        if clean_text:
            chunks.append({
                "id": f"{file_path.stem}_page_{idx}",
                "source": file_name,
                "file_type": "pdf",
                "page_or_slide": idx,
                "topic": topic,
                "text": clean_text
            })

    logger.info(f"Extraídas {len(chunks)} páginas con texto de {file_name}")
    return chunks


def extract_text_from_txt_or_md(file_path: Path) -> List[Dict[str, Any]]:
    """Soporte para archivos de texto o transcripciones Markdown."""
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    file_name = file_path.name
    topic = file_path.stem.replace("_", " ").title()

    sections = [s.strip() for s in text.split("\n## ") if s.strip()]
    chunks = []

    for idx, sec in enumerate(sections, start=1):
        clean_sec = sec.strip()
        if len(clean_sec) < 25:
            continue
        chunks.append({
            "id": f"{file_path.stem}_section_{idx}",
            "source": file_name,
            "file_type": file_path.suffix.replace(".", ""),
            "page_or_slide": idx,
            "topic": topic,
            "text": f"## {clean_sec}" if not clean_sec.startswith("#") else clean_sec
        })

    logger.info(f"Extraídas {len(chunks)} secciones de {file_name}")
    return chunks


def process_all_documents(raw_dir: Path = RAW_DATA_DIR, output_file: Path = CHUNKS_FILE) -> List[Dict[str, Any]]:
    """Recorre la carpeta raw/ y procesa todos los documentos soportados con tablas y notas."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    supported_extensions = {
        ".pptx": extract_text_from_pptx,
        ".pdf": extract_text_from_pdf,
        ".txt": extract_text_from_txt_or_md,
        ".md": extract_text_from_txt_or_md
    }

    files = [f for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]

    if not files:
        logger.warning(f"No se encontraron archivos en {raw_dir}.")
        return []

    logger.info(f"Procesando {len(files)} archivos en {raw_dir} (extrayendo textos, tablas y notas)...")

    for file_path in files:
        extractor = supported_extensions[file_path.suffix.lower()]
        try:
            chunks = extractor(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error procesando {file_path.name}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Total de fragmentos guardados en {output_file}: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    process_all_documents()
