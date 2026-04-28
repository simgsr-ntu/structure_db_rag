import plotly.express as px
import plotly.io as pio
import sqlite3
import os
import uuid
from langchain_core.tools import tool

# Set dark theme by default for Plotly
pio.templates.default = "plotly_dark"

def make_viz_tool(registry):
    db_path = registry.db_path

    @tool
    def viz_tool(chart_name: str) -> str:
        """Generates an interactive Plotly chart from live sermon data and returns the JSON file path.
        Supported chart_name values:
        - 'sermons_per_speaker' — bar chart of sermon count per speaker (top 15)
        - 'sermons_per_year' — bar chart of sermon count per year
        - 'verses_per_book' — bar chart of most-preached Bible books from verses table (top 15)
        - 'sermons_scatter' — bubble chart of sermon count by speaker and year
        Returns the file path to the saved Plotly JSON."""
        
        try:
            with sqlite3.connect(db_path) as conn:
                if chart_name == "sermons_per_speaker":
                    rows = conn.execute(
                        "SELECT speaker, COUNT(*) as n FROM sermons "
                        "WHERE speaker IS NOT NULL AND speaker != '' "
                        "GROUP BY speaker ORDER BY n DESC LIMIT 15"
                    ).fetchall()
                    if not rows:
                        return "No sermon data found."
                    
                    speakers, counts = zip(*rows)
                    fig = px.bar(
                        x=counts, y=speakers, orientation='h',
                        title="Top 10 Speakers by Sermon Count",
                        labels={'x': 'Number of Sermons', 'y': 'Speaker'},
                        color=counts, color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)

                elif chart_name == "sermons_per_year":
                    rows = conn.execute(
                        "SELECT year, COUNT(*) as n FROM sermons "
                        "WHERE year IS NOT NULL "
                        "GROUP BY year ORDER BY year"
                    ).fetchall()
                    if not rows:
                        return "No sermon data found."
                    
                    years, counts = zip(*rows)
                    fig = px.bar(
                        x=[str(y) for y in years], y=counts,
                        title="Sermons per Year",
                        labels={'x': 'Year', 'y': 'Number of Sermons'},
                        color=counts, color_continuous_scale='Viridis'
                    )

                elif chart_name == "verses_per_book":
                    rows = conn.execute(
                        "SELECT book, COUNT(*) as n FROM verses "
                        "WHERE book IS NOT NULL AND book != '' "
                        "GROUP BY book ORDER BY n DESC LIMIT 15"
                    ).fetchall()
                    if not rows:
                        return "No verse data found. Run ingest.py first."
                    books, counts = zip(*rows)
                    fig = px.bar(
                        x=counts, y=books, orientation='h',
                        title="Top 15 Preached Bible Books",
                        labels={'x': 'Times Preached', 'y': 'Bible Book'},
                        color=counts, color_continuous_scale='Greens'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)

                elif chart_name == "sermons_scatter":
                    rows = conn.execute(
                        "SELECT COALESCE(year, CAST(SUBSTR(date, 1, 4) AS INTEGER)) as yr, "
                        "speaker, COUNT(*) as n FROM sermons "
                        "WHERE (year IS NOT NULL OR date IS NOT NULL) "
                        "AND speaker IS NOT NULL AND speaker != '' "
                        "GROUP BY yr, speaker ORDER BY yr"
                    ).fetchall()
                    if not rows:
                        return "No sermon data found."

                    years = [str(r[0]) for r in rows]
                    speakers = [r[1] for r in rows]
                    counts = [r[2] for r in rows]

                    all_years = sorted({str(r[0]) for r in rows})

                    fig = px.scatter(
                        x=years, y=speakers, size=counts, color=counts,
                        title="Sermon Count by Speaker and Year",
                        labels={'x': 'Year', 'y': 'Speaker', 'size': 'Count'},
                        hover_name=speakers, size_max=16,
                        color_continuous_scale='Plasma',
                        category_orders={'x': all_years},
                    )
                    fig.update_xaxes(type='category', tickmode='array', tickvals=all_years, tickangle=-45)

                else:
                    return (
                        f"Unknown chart '{chart_name}'. "
                        "Valid options: sermons_per_speaker, sermons_per_year, "
                        "verses_per_book, sermons_scatter."
                    )

            left_margin = 180 if chart_name in ("sermons_per_speaker", "verses_per_book", "sermons_scatter") else 60
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Inter, sans-serif",
                title_font_size=18,
                margin=dict(l=left_margin, r=40, t=60, b=40),
            )

            file_path = os.path.join("/tmp", f"bbtc_chart_{uuid.uuid4().hex[:8]}.json")
            fig.write_json(file_path)
            return file_path

        except Exception as e:
            return f"Chart generation error: {e}"

    return viz_tool
