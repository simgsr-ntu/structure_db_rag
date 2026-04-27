import os
import json
from dotenv import load_dotenv
from src.ingestion.bible.epub_parser import BibleEpubParser
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm

load_dotenv()

def generate_verse_meaning(llm, verse_obj):
    """Generates a brief theological meaning for the verse."""
    prompt = f"Explain the meaning of {verse_obj['reference']} in the context of the book of {verse_obj['book']}. Keep it concise (2-3 sentences)."
    try:
        # Using a very lightweight call
        res = llm.invoke(prompt)
        return res.content.strip()
    except:
        return "Explanation pending research."

def ingest_bible(version_id: str, filename: str, logger=print):
    bible_dir = "data/bibles"
    filepath = os.path.join(bible_dir, filename)
    
    if not os.path.exists(filepath):
        logger(f"❌ Skipping {filename} (not found at {filepath})")
        return False

    store = SermonVectorStore()
    
    logger(f"📖 --- Processing Bible Version: {version_id} ---")
    parser = BibleEpubParser(filepath, version_id)
    verses = parser.parse()
    logger(f"✅ Extracted {len(verses)} verses from {version_id}")

    # Batching for ChromaDB
    BATCH_SIZE = 200
    for i in range(0, len(verses), BATCH_SIZE):
        batch = verses[i:i+BATCH_SIZE]
        
        chunks = []
        metadatas = []
        ids = []

        for v in batch:
            v['verse_meaning'] = "Meaning generation disabled for bulk ingest to save costs."
            
            chunks.append(v['text'])
            ids.append(f"{v['version']}_{v['ref_id']}")
            metadatas.append({
                "book": v['book'],
                "chapter": v['chapter'],
                "verse": v['verse'],
                "version": v['version'],
                "reference": v['reference'],
                "ref_id": v['ref_id'],
                "verse_meaning": v['verse_meaning']
            })
        
        store.upsert_bible_chunks(chunks, metadatas, ids)
        if i % (BATCH_SIZE * 5) == 0:
            logger(f"📡 Uploaded {i + len(batch)}/{len(verses)} verses...")

    return True

def main():
    # Setup paths
    files = {
        "NIV.epub": "NIV",
        "ESV The Holy Bible.epub": "ESV",
        "Bible - American Standard Version.epub": "ASV"
    }

    for filename, version in files.items():
        ingest_bible(version, filename)

if __name__ == "__main__":
    main()
