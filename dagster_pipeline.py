import os
from datetime import datetime
from dagster import asset, Definitions, ScheduleDefinition, AssetSelection, define_asset_job, AssetExecutionContext, MetadataValue
from src.scraper.bbtc_scraper import BBTCScraper
from src.storage.sqlite_store import SermonRegistry
from src.storage.chroma_store import SermonVectorStore
from src.ingestion.metadata_extractor import MetadataExtractor

# Initialize components
registry = SermonRegistry()
vector_store = SermonVectorStore()
extractor = MetadataExtractor()
scraper = BBTCScraper(registry=registry)

@asset
def sermon_ingestion_summary(context: AssetExecutionContext):
    """
    Weekly asset that scrapes new sermons for the current year,
    extracts metadata, and updates the vector store.
    """
    current_year = datetime.now().year
    years_to_scrape = range(2015, current_year + 1)
    
    context.log.info(f"🚀 Starting ingestion for years: {list(years_to_scrape)}")
    
    for year in years_to_scrape:
        context.log.info(f"📅 Scrapping year {year}...")
        # 1. Scrape and download (this updates SQLite and saves .txt files)
        scraper.scrape_year(year)
    
    # 2. Process newly 'extracted' sermons to add metadata and vector embeddings
    sermons = registry.get_all_sermons()
    pending_sermons = [s for s in sermons if s['status'] in ('extracted', 'processed')]
    total_to_process = len(pending_sermons)
    
    context.log.info(f"📋 Found {total_to_process} sermons pending indexing.")
    
    processed_count = 0
    for i, sermon in enumerate(pending_sermons):
        progress = f"[{i+1}/{total_to_process}]"
        sermon_id = sermon['sermon_id']
        txt_filename = os.path.splitext(sermon['filename'])[0] + ".txt"
        txt_path = os.path.join("data/sermons", txt_filename)
        
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract metadata using LLM
            context.log.info(f"🧠 {progress} Extracting metadata for {sermon['filename']}...")
            metadata = extractor.extract(content[:500])
            
            # Update SQLite with metadata
            update_record = {**sermon, **metadata, "status": "indexed"}
            registry.insert_sermon(update_record)
            
            # Upsert to ChromaDB
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(content)
            
            if not chunks:
                chunks = [content[:500]]
            
            context.log.info(f"📡 {progress} Vectorizing {sermon['filename']} ({len(chunks)} chunks)...")
            metadatas = [update_record] * len(chunks)
            ids = [f"{sermon_id}_{i}" for i in range(len(chunks))]
            vector_store.upsert_sermon_chunks(chunks, metadatas, ids)
            
            processed_count += 1
        else:
            context.log.warning(f"⚠️ {progress} Text file not found for {sermon['filename']}")
    
    # Add final metadata for the UI
    context.add_output_metadata(
        metadata={
            "year": current_year,
            "newly_indexed": MetadataValue.int(processed_count),
            "total_in_db": MetadataValue.int(len(sermons)),
            "last_run": MetadataValue.text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
    )
    
    return {
        "year": current_year,
        "newly_indexed": processed_count,
        "total_in_db": len(sermons)
    }

# Job definition
ingestion_job = define_asset_job("sermon_ingestion_job", selection=AssetSelection.all())

# Weekly schedule (Sunday at midnight)
sermon_weekly_schedule = ScheduleDefinition(
    job=ingestion_job,
    cron_schedule="0 0 * * 0", 
)

defs = Definitions(
    assets=[sermon_ingestion_summary],
    schedules=[sermon_weekly_schedule],
    jobs=[ingestion_job],
)
