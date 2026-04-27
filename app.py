import gradio as gr
import os
import subprocess
import time
import urllib.request
from dotenv import load_dotenv
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm
from src.ui_helpers import extract_chart_path, fetch_archive_stats, render_stats_bar
from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool
from src.tools.vector_tool import make_vector_tool
from langchain_core.messages import HumanMessage, AIMessage
from src.tools.bible_tool import make_bible_tool
from src.tools.viz_tool import make_viz_tool
import plotly.io as pio
from langchain.agents import create_agent

load_dotenv()

# Ensure Plotly uses dark template for consistency
pio.templates.default = "plotly_dark"

def _ensure_ollama(timeout: int = 20) -> bool:
    def _is_up() -> bool:
        try:
            urllib.request.urlopen("http://127.0.0.1:11434", timeout=2)
            return True
        except Exception:
            return False

    if _is_up():
        return True

    print("🦙 Ollama not running — starting it now...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1)
        if _is_up():
            print("🦙 Ollama is ready.")
            return True

    print("⚠️  Ollama did not start within the timeout.")
    return False
_ensure_ollama()

try:
    registry = SermonRegistry()
    vector_store = SermonVectorStore()
    llm = get_llm(temperature=0.1)

    sql_tool = make_sql_tool(registry)
    vector_tool = make_vector_tool(vector_store)
    bible_tool = make_bible_tool(vector_store)
    viz_tool = make_viz_tool(registry)

    SYSTEM_PROMPT = (
        "You are the BBTC Sermon Intelligence Assistant for Bethesda Bedok-Tampines Church.\n\n"
        "## Tool routing\n"
        "- Use 'sql_query_tool' for: counts, statistics, lists of speakers/years, date lookups, "
        "questions that need numbers (e.g. 'how many sermons', 'top 5 speakers').\n"
        "- Use 'search_sermons_tool' for: questions about sermon *content*, topics, theology, "
        "what a pastor said, summaries of specific sermons. Pass 'year' or 'speaker' filters "
        "when the user specifies them.\n"
        "- For 'what was said about X in year Y' or 'what did speaker Z say about X', use search_sermons_tool "
        "with the year/speaker filter directly — do not run sql_query_tool first.\n"
        "- Use 'compare_bible_versions' only when the user explicitly asks to compare Bible translations.\n"
        "- Use 'viz_tool' only when the user asks for a chart or visualization. "
        "Valid chart_name values: 'sermons_per_speaker', 'sermons_per_year', 'top_bible_books', 'sermons_scatter'. "
        "When viz_tool returns a file path, copy that exact path into your response verbatim — do not describe or summarise the chart data.\n\n"
        "## Grounding rules\n"
        "- Answer ONLY from data returned by the tools. Never invent sermon content, speaker names, "
        "dates, or verses.\n"
        "- When answering from search_sermons_tool results, cite the sermon filename and speaker name for every excerpt quoted.\n"
        "- If the tools return no relevant data, say so explicitly — do not guess or fill gaps.\n"
        "- If you need more information to answer precisely, call the relevant tool again with "
        "a refined query before responding.\n"
    )

    agent = create_agent(llm, tools=[sql_tool, vector_tool, bible_tool, viz_tool], system_prompt=SYSTEM_PROMPT)

except Exception as e:
    print(f"⚠️ Initialization warning: {e}")
    agent = None
    registry = None
    vector_store = None

_stats_bar_html = (
    render_stats_bar(fetch_archive_stats(registry.db_path))
    if registry is not None
    else render_stats_bar(None)
)


def respond(message, history):
    if agent is None:
        return "⚠️ Agent not initialized. Check that Ollama is running."

    truncated_history = history[-6:] if len(history) > 6 else history
    messages = []
    for turn in truncated_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            content = turn["content"]
            if isinstance(content, list):
                # Handle complex content (text + plot)
                text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
                content = " ".join(text_parts)
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    try:
        result = agent.invoke({"messages": messages})
        final = result["messages"][-1].content
        if not isinstance(final, str):
            final = str(final)

        # If the LLM dropped the chart path from its response, recover it from ToolMessage
        if "/tmp/bbtc_chart_" not in final:
            import re
            for msg in result["messages"]:
                match = re.search(r'/tmp/bbtc_chart_[a-f0-9]+\.(png|json)', str(msg.content))
                if match:
                    final = final.rstrip() + "\n" + match.group(0)
                    break

        return final
    except Exception as e:
        return f"⚠️ An error occurred while processing your request: {e}"


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

footer {visibility: hidden}
body { background-color: #020617; }

.gradio-container {
    background-color: #020617 !important;
    color: #f8fafc;
    font-family: 'Outfit', sans-serif !important;
    max-width: 1400px !important;
}

.sidebar {
    background: rgba(30, 41, 59, 0.3) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 24px !important;
    border-radius: 20px;
}

.chatbot-container {
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: rgba(15, 23, 42, 0.6) !important;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    overflow: hidden !important;
}

.message-user {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 14px 20px !important;
    color: white !important;
    font-weight: 500;
}

.message-assistant {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 20px 20px 20px 4px !important;
    padding: 14px 20px !important;
}

.input-container {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    margin-top: 10px !important;
}

.btn-primary {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-radius: 12px !important;
    transition: all 0.3s ease;
}
.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 20px -5px rgba(59, 130, 246, 0.4);
}

#title-container {
    margin-bottom: 40px;
    text-align: left;
    display: flex;
    align-items: center;
    gap: 25px;
}
#title-container img {
    height: 70px;
    filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.3));
}
#title-text h1 {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
#title-text p {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 400;
    margin-top: 4px;
}

.stats-bar {
    display: flex;
    justify-content: space-between;
    background: linear-gradient(to right, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 14px;
    padding: 12px 24px;
    margin-bottom: 24px;
    color: #cbd5e1;
    font-size: 0.95rem;
}

.status-badge {
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.status-online { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.2); }
.status-offline { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }
"""

with gr.Blocks(title="BBTC Sermon Intelligence") as demo:
    with gr.Row(elem_id="header"):
        with gr.Column(scale=4):
            gr.HTML("""
                <div id='title-container'>
                    <img src='https://www.bbtc.com.sg/wp-content/uploads/2021/04/BBTC-Logo-Header.png' alt='Logo'>
                    <div id='title-text'>
                        <h1>Sermon Intelligence</h1>
                        <p>Hybrid Agentic RAG Platform • Professional Edition</p>
                    </div>
                </div>
            """)

    gr.HTML(_stats_bar_html)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                min_height=550,
                show_label=False,
                elem_classes="chatbot-container",
                avatar_images=(None, "https://www.bbtc.com.sg/wp-content/uploads/2021/04/BBTC-Logo-Header.png")
            )
            with gr.Row(elem_classes="input-container"):
                msg = gr.Textbox(
                    placeholder="Describe the data you need or ask a theological question...",
                    container=False,
                    scale=7,
                )
                submit = gr.Button("🚀 Execute", variant="primary", scale=1, elem_classes="btn-primary")

            gr.Examples(
                examples=[
                    ["Show an interactive chart of how many sermons were preached each year"],
                    ["Create an interactive bar chart of sermon count per speaker"],
                    ["Show a scatter plot of sermon counts by speaker and year"],
                    ["How many sermons are in the archive and who are the top 5 speakers?"],
                    ["What was the most recent sermon and what were its key points?"],
                    ["What have our pastors said about faith during trials and suffering?"],
                ],
                inputs=msg,
                label="💡 Strategic Inquiries"
            )

        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### 🛠️ System Health")

            vec_status = "online" if vector_store else "offline"
            ollama_status = "online" if (vector_store and vector_store._embeddings is not None) else "offline"
            gr.HTML(f"""
                <div style='display: flex; flex-direction: column; gap: 12px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #94a3b8;'>Vector Store</span>
                        <span class='status-badge status-{vec_status}'>{vec_status}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #94a3b8;'>SQL Registry</span>
                        <span class='status-badge status-online'>active</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: #94a3b8;'>Inference</span>
                        <span class='status-badge status-{ollama_status}'>ollama</span>
                    </div>
                </div>
            """)

            gr.Markdown("---")
            gr.Markdown("### 🎯 Capabilities")
            gr.Markdown(
                "- **Semantic Search**: Retrieval of sermon content across a decade of archives.\n"
                "- **SQL Analytics**: High-precision metadata querying for statistics and counts.\n"
                "- **Dynamic Viz**: Real-time generation of interactive Plotly visualizations.\n"
                "- **Bible Context**: Multi-version Bible referencing and cross-comparison."
            )

            gr.Markdown("---")
            clear = gr.Button("🗑️ Reset Workspace", variant="secondary")

    def user_msg(user_message, history: list):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    def bot_msg(history: list):
        if not history or history[-1]["role"] != "user":
            return history

        user_message = history[-1]["content"]
        chat_history = history[:-1]
        bot_message = respond(user_message, chat_history)

        text, chart_path = extract_chart_path(bot_message)
        
        content = []
        if text:
            content.append({"type": "text", "text": text})
            
        if chart_path:
            if chart_path.endswith('.json'):
                try:
                    import plotly.io as pio
                    fig = pio.read_json(chart_path)
                    content.append(gr.Plot(fig))
                except Exception as e:
                    content.append({"type": "text", "text": f"\n⚠️ Error loading interactive chart: {e}"})
            else:
                content.append({"type": "image", "image": {"path": chart_path}})

        if not content:
            content = bot_message

        history.append({"role": "assistant", "content": content})
        return history

    disable_submit = lambda: gr.update(value="⏳ Processing...", interactive=False)
    enable_submit = lambda: gr.update(value="🚀 Execute", interactive=True)

    msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot], chatbot
    ).then(
        enable_submit, None, submit
    )
    submit.click(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot], chatbot
    ).then(
        enable_submit, None, submit
    )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=custom_css,
        allowed_paths=["/tmp"]
    )
