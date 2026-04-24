import sqlite3
from langchain_core.tools import tool

def make_sql_tool(registry):
    db_path = registry.db_path

    @tool
    def sql_query_tool(query: str) -> str:
        """Executes a SQL query against the sermons database and returns the result."""
        print(f"DEBUG: Executing SQL: {query}")
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(query)
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                if not rows:
                    return "No results found."
                
                # Format as a string
                result = f"Columns: {', '.join(columns)}\n"
                for row in rows:
                    result += f"{row}\n"
                return result
        except Exception as e:
            schema = "sermons(sermon_id, filename, speaker, bible_book, primary_verse, year)"
            return f"Error: {str(e)}. Please check your column names. The schema is: {schema}"

    return sql_query_tool
