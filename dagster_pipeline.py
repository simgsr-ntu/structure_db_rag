"""
Dagster pipeline — thin wrapper around ingest.py.
Weekly schedule: Saturday at 22:00 (so new weekend files are ready).

UI:  DAGSTER_HOME=$(mktemp -d) dagster dev -m dagster_pipeline
Run: dagster asset materialize --select sermon_ingestion -m dagster_pipeline
"""

from dagster import (
    asset, Definitions, ScheduleDefinition, AssetSelection,
    define_asset_job, AssetExecutionContext, MetadataValue, in_process_executor,
)
from ingest import run_pipeline


@asset
def sermon_ingestion(context: AssetExecutionContext):
    """Weekly incremental ingestion of new BBTC sermons."""
    context.log.info("Starting incremental sermon ingestion...")
    run_pipeline(wipe=False, year=None, incremental=True)
    context.log.info("Ingestion complete.")
    return MetadataValue.text("done")


ingestion_job = define_asset_job(
    "sermon_ingestion_job",
    selection=AssetSelection.assets(sermon_ingestion),
    executor_def=in_process_executor,
)

sermon_weekly_schedule = ScheduleDefinition(
    job=ingestion_job,
    cron_schedule="0 22 * * 6",  # Saturday 22:00
)

defs = Definitions(
    assets=[sermon_ingestion],
    schedules=[sermon_weekly_schedule],
    jobs=[ingestion_job],
    executor=in_process_executor,
)
