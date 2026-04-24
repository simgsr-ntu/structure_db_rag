import os
import sqlite3
from src.storage.chroma_store import SermonVectorStore
from src.storage.sqlite_store import SermonRegistry

def quick_ingest():
    registry = SermonRegistry()
    vector_store = SermonVectorStore()
    
    print("🚀 Starting quick ingestion of existing sermons...")
    
    sermons = registry.get_all_sermons()
    count = 0
    
    for sermon in sermons:
        # We'll ingest anything that is 'processed' or 'extracted'
        if sermon['status'] in ('processed', 'extracted'):
            sermon_id = sermon['sermon_id']
            txt_filename = os.path.splitext(sermon['filename'])[0] + ".txt"
            txt_path = os.path.join("data/sermons", txt_filename)
            
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                print(f"Indexing {sermon['filename']}...")
                
                # Use existing metadata if available, otherwise minimal
                metadata = {
                    "filename": sermon['filename'],
                    "year": sermon['year'],
                    "language": sermon['language'],
                    "speaker": sermon.get('speaker', 'Unknown'),
                    "series": sermon.get('series', 'Unknown')
                }
                
                # Simple chunking by paragraphs for better retrieval
                chunks = [p for p in content.split("\n\n") if len(p.strip()) > 50]
                if not chunks:
                    chunks = [content]
                    
                metadatas = [metadata] * len(chunks)
                ids = [f"{sermon_id}_{i}" for i in range(len(chunks))]
                
                try:
                    vector_store.upsert_sermon_chunks(chunks, metadatas, ids)
                    # Update status to 'indexed'
                    registry.mark_processed(sermon['url'], status='indexed')
                    count += 1
                except Exception as e:
                    print(f"❌ Failed to index {sermon['filename']}: {e}")
                    print("💡 Make sure Ollama is running or configure an alternative embedding model.")
                    return
            else:
                print(f"⚠️ Text file not found: {txt_path}")
                
    print(f"✅ Quick ingestion complete. {count} sermons indexed.")

if __name__ == "__main__":
    quick_ingest()
