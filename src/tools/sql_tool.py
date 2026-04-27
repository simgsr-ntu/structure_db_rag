import sqlite3
from langchain_core.tools import tool

def make_sql_tool(registry):
    db_path = registry.db_path

    @tool
    def sql_query_tool(query: str) -> str:
        """Executes a SQL query against the church database.
        Tables:
        - sermons: filename, url, speaker, date, series, bible_book, primary_verse, language, file_type, year, status.
        - sermon_intelligence: sermon_id, speaker (normalized), primary_verse, verses_used (comma-separated), summary.
        Use JOIN on sermon_id if needed. Returns up to 50 rows."""
        print(f"DEBUG: Executing SQL: {query}")
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(query)
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()[:50]
                if not rows:
                    return "No results found."

                result = f"Columns: {', '.join(columns)}\n"
                for row in rows:
                    result += f"{row}\n"
                return result
        except Exception as e:
            schema = (
                "Tables:\n"
                "1. sermons(sermon_id, filename, url, speaker, date, series, bible_book, primary_verse, language, file_type, year, status)\n"
                "2. sermon_intelligence(sermon_id, speaker, primary_verse, verses_used, summary)"
            )
            return f"SQL Error: {str(e)}. Full schema:\n{schema}"

    return sql_query_tool
