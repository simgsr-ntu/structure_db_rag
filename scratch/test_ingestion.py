import os
from dagster import AssetExecutionContext
from dagster_pipeline import bible_ingestion

class MockContext:
    def __init__(self):
        self.log = self
    def info(self, msg):
        print(f"INFO: {msg}")
    def add_output_metadata(self, metadata):
        print(f"METADATA: {metadata}")

ctx = MockContext()
try:
    bible_ingestion(ctx)
except Exception as e:
    import traceback
    traceback.print_exc()
