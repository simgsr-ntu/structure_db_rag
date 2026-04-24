import matplotlib.pyplot as plt
import sqlite3
import os
import uuid
from langchain_core.tools import tool

def make_matplotlib_tool(registry):
    db_path = registry.db_path

    @tool
    def matplotlib_tool(chart_name: str) -> str:
        """Generates a visualization based on the requested chart_name and returns the file path."""
        # This is a simplified version that creates dummy charts for demonstration
        # based on the chart_name provided. In a real app, you'd query the DB.
        
        plt.figure(figsize=(10, 6))
        
        if chart_name == "sermons_per_pastor_by_year":
            plt.title("Sermons per Pastor by Year")
            plt.bar(["Pastor A", "Pastor B", "Pastor C"], [10, 15, 8])
        elif chart_name == "bible_book_coverage":
            plt.title("Bible Book Coverage")
            plt.pie([40, 30, 20, 10], labels=["Genesis", "Psalms", "John", "Romans"])
        else:
            plt.title(f"Chart: {chart_name}")
            plt.text(0.5, 0.5, "Visualization data would appear here", ha='center')

        file_name = f"bbtc_chart_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join("/tmp", file_name)
        plt.savefig(file_path)
        plt.close()
        
        return file_path

    return matplotlib_tool
