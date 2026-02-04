"""
SCALPEL Textual TUI.

Scientific Critique & Analysis Pipeline for Evidence Literature.
A terminal-based graphical interface using Textual.
"""

from pathlib import Path
from typing import Iterator

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    OptionList,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker, WorkerState

from scalpel import __version__
from scalpel.config import settings


ANALYSIS_TYPES = [
    ("bullshit_score", "🔪 Bullshit Score"),
    ("summarize", "📝 Summary"),
    ("critique_methodology", "🔬 Methodology Critique"),
    ("critique_statistics", "📊 Statistics Critique"),
    ("critique_full", "📖 Full Critique"),
    ("extract_claims", "💡 Extract Claims"),
    ("identify_limitations", "⚠️ Limitations"),
    ("quick_assess", "🎯 Quick Assessment"),
]


class ScalpelApp(App):
    """SCALPEL Terminal User Interface."""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #sidebar {
        width: 32;
        background: $panel;
        border-right: solid $primary;
        padding: 1;
    }
    
    #analysis-type {
        height: 12;
        max-height: 12;
    }
    
    #run-btn {
        margin-top: 1;
        width: 100%;
    }
    
    #main {
        padding: 1;
    }
    
    #paper-info {
        height: auto;
        max-height: 10;
        background: $boost;
        border: round $primary;
        padding: 1;
        margin-bottom: 1;
    }
    
    #analysis-output {
        border: round $accent;
        padding: 1;
    }
    
    .section-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    #file-input {
        margin-bottom: 1;
    }
    
    #library-list {
        height: 1fr;
    }
    
    #search-results {
        height: 1fr;
        border: round $accent;
        padding: 1;
    }
    
    .score-good {
        color: green;
    }
    
    .score-warning {
        color: yellow;
    }
    
    .score-bad {
        color: red;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("a", "switch_tab('analyze')", "Analyze"),
        Binding("l", "switch_tab('library')", "Library"),
        Binding("s", "switch_tab('search')", "Search"),
        Binding("c", "switch_tab('config')", "Config"),
    ]
    
    def __init__(self):
        super().__init__()
        self.current_paper = None
        self.library_papers: list[dict] = []
        self.title = "SCALPEL"
        self.sub_title = f"v{__version__}"
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with TabbedContent(id="tabs"):
            with TabPane("Analyze", id="analyze"):
                yield from self._compose_analyze_tab()
            
            with TabPane("Library", id="library"):
                yield from self._compose_library_tab()
            
            with TabPane("Search", id="search"):
                yield from self._compose_search_tab()
            
            with TabPane("Config", id="config"):
                yield from self._compose_config_tab()
        
        yield Footer()
    
    def _compose_analyze_tab(self) -> ComposeResult:
        with Horizontal():
            with VerticalScroll(id="sidebar"):
                yield Label("📄 Load Paper", classes="section-title")
                yield Input(placeholder="Path to PDF...", id="file-input")
                yield Button("Load PDF", id="load-btn", variant="primary")
                yield Label("arXiv:", classes="section-title")
                yield Input(placeholder="1706.03762", id="arxiv-input")
                yield Button("Fetch arXiv", id="arxiv-btn")
                yield Label("🔬 Analysis", classes="section-title")
                yield OptionList(
                    *[item[1] for item in ANALYSIS_TYPES],
                    id="analysis-type",
                )
                yield Button("🚀 Run", id="run-btn", variant="success")
            
            with Vertical(id="main"):
                yield Static("No paper loaded", id="paper-info")
                with VerticalScroll(id="analysis-output"):
                    yield Markdown("", id="output-md")
    
    def _compose_library_tab(self) -> ComposeResult:
        with Vertical():
            yield Label("📚 Paper Library", classes="section-title")
            with Horizontal():
                yield Button("🔄 Refresh", id="refresh-lib")
                yield Button("🗑️ Remove Selected", id="remove-paper", variant="error")
            yield ListView(id="library-list")
    
    def _compose_search_tab(self) -> ComposeResult:
        with Vertical():
            yield Label("🔍 Search Papers", classes="section-title")
            with Horizontal():
                yield Input(placeholder="Enter search query...", id="search-input")
                yield Button("Search", id="search-btn", variant="primary")
            with VerticalScroll(id="search-results"):
                yield Markdown("", id="search-md")
    
    def _compose_config_tab(self) -> ComposeResult:
        with Vertical():
            yield Label("⚙️ Configuration", classes="section-title")
            yield Markdown(self._get_config_md(), id="config-md")
    
    def _get_config_md(self) -> str:
        return f"""
## Ollama Settings
- **Host:** {settings.ollama_host}
- **LLM Model:** {settings.ollama_model}
- **Embedding Model:** {settings.embedding_model}
- **Temperature:** {settings.model_temperature}
- **Context Length:** {settings.model_context_length:,}

## Storage Settings
- **Papers Directory:** {settings.papers_dir}
- **Vector DB:** {settings.lancedb_path}
- **Chunk Size:** {settings.chunk_size}
- **Chunk Overlap:** {settings.chunk_overlap}

---
*Edit `.env` file to change settings*
"""
    
    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab."""
        self.query_one("#tabs", TabbedContent).active = tab_id
    
    @on(Button.Pressed, "#load-btn")
    def load_pdf(self) -> None:
        """Load a PDF file."""
        file_input = self.query_one("#file-input", Input)
        path_str = file_input.value.strip()
        
        if not path_str:
            self.notify("Please enter a file path", severity="warning")
            return
        
        path = Path(path_str)
        
        if not path.exists():
            self.notify(f"File not found: {path}", severity="error")
            return
        
        self._do_load_pdf(path)
    
    @work(exclusive=True, thread=True)
    def _do_load_pdf(self, path: Path) -> None:
        """Load PDF in background thread."""
        from scalpel.ingestion import extract_pdf
        from scalpel.embeddings import add_paper
        
        self.call_from_thread(self.notify, "Loading PDF...")
        
        try:
            paper = extract_pdf(path)
            self.current_paper = paper
            self.call_from_thread(self._update_paper_info)
            
            self.call_from_thread(self.notify, "Adding to library...")
            count = add_paper(paper, verbose=False)
            if count > 0:
                self.call_from_thread(
                    self.notify, f"Loaded & indexed: {paper.title} ({count} chunks)", severity="information"
                )
            else:
                self.call_from_thread(
                    self.notify, f"Loaded: {paper.title} (already in library)", severity="information"
                )
            self.call_from_thread(self.refresh_library)
        except Exception as e:
            self.call_from_thread(
                self.notify, f"Error loading PDF: {e}", severity="error"
            )
    
    @on(Button.Pressed, "#arxiv-btn")
    def fetch_arxiv(self) -> None:
        """Fetch paper from arXiv."""
        arxiv_input = self.query_one("#arxiv-input", Input)
        arxiv_id = arxiv_input.value.strip()
        
        if not arxiv_id:
            self.notify("Please enter an arXiv ID", severity="warning")
            return
        
        self._do_fetch_arxiv(arxiv_id)
    
    @work(exclusive=True, thread=True)
    def _do_fetch_arxiv(self, arxiv_id: str) -> None:
        """Fetch from arXiv in background thread."""
        from scalpel.ingestion import fetch_arxiv
        from scalpel.embeddings import add_paper
        
        self.call_from_thread(self.notify, "Fetching from arXiv...")
        
        try:
            paper = fetch_arxiv(arxiv_id)
            self.current_paper = paper
            self.call_from_thread(self._update_paper_info)
            
            self.call_from_thread(self.notify, "Adding to library...")
            count = add_paper(paper, verbose=False)
            if count > 0:
                self.call_from_thread(
                    self.notify, f"Loaded & indexed: {paper.title} ({count} chunks)", severity="information"
                )
            else:
                self.call_from_thread(
                    self.notify, f"Loaded: {paper.title} (already in library)", severity="information"
                )
            self.call_from_thread(self.refresh_library)
        except Exception as e:
            self.call_from_thread(
                self.notify, f"Error fetching: {e}", severity="error"
            )
    
    def _update_paper_info(self) -> None:
        """Update the paper info display."""
        info_widget = self.query_one("#paper-info", Static)
        
        if self.current_paper is None:
            info_widget.update("No paper loaded")
            return
        
        paper = self.current_paper
        
        authors_str = "Unknown"
        if paper.metadata:
            authors = paper.metadata.get("author", "")
            if authors:
                authors_str = authors[:50] + "..." if len(authors) > 50 else authors
        
        info = f"📄 **{paper.title}**\n"
        info += f"Authors: {authors_str}\n"
        info += f"Sections: {len(paper.sections)} | Characters: {len(paper.text):,}"
        
        info_widget.update(info)
    
    @on(Button.Pressed, "#run-btn")
    def run_analysis(self) -> None:
        """Run the selected analysis."""
        if self.current_paper is None:
            self.notify("Please load a paper first", severity="warning")
            return
        
        option_list = self.query_one("#analysis-type", OptionList)
        if option_list.highlighted is None:
            self.notify("Please select an analysis type", severity="warning")
            return
        
        analysis_key = ANALYSIS_TYPES[option_list.highlighted][0]
        self._do_analysis(analysis_key)
    
    @work(exclusive=True, thread=True)
    def _do_analysis(self, analysis_key: str) -> None:
        """Run analysis in background thread."""
        from scalpel.analysis import Analyzer, BullshitScore
        from scalpel.analysis.llm_client import LLMClient
        from scalpel.analysis.prompts import get_template
        from scalpel.ingestion import chunk_paper
        
        output_md = self.query_one("#output-md", Markdown)
        
        self.call_from_thread(output_md.update, "*Initializing analysis...*")
        
        client = LLMClient()
        
        if not client.is_available():
            self.call_from_thread(
                output_md.update,
                f"❌ **Error:** Model '{client.model}' not available. Is Ollama running?"
            )
            return
        
        paper = self.current_paper
        
        if analysis_key == "bullshit_score":
            self.call_from_thread(output_md.update, "*Calculating bullshit score...*")
            analyzer = Analyzer(client=client)
            score = analyzer.bullshit_score(paper, verbose=False)
            
            color_class = (
                "score-good" if score.overall_score <= 4
                else "score-warning" if score.overall_score <= 6
                else "score-bad"
            )
            
            result = f"# 🔪 Bullshit Score: {score.overall_score}/10 — {score.rating}\n\n"
            result += f"{score.summary}\n\n"
            
            if score.red_flags:
                result += "## Red Flags\n"
                for flag in score.red_flags:
                    result += f"- {flag}\n"
            
            result += "\n## Component Scores\n"
            if score.methodology_score is not None:
                result += f"- **Methodology:** {score.methodology_score}/10\n"
            if score.statistics_score is not None:
                result += f"- **Statistics:** {score.statistics_score}/10\n"
            if score.claims_score is not None:
                result += f"- **Claims:** {score.claims_score}/10\n"
            if score.transparency_score is not None:
                result += f"- **Transparency:** {score.transparency_score}/10\n"
            
            self.call_from_thread(output_md.update, result)
        else:
            template = get_template(analysis_key)
            self.call_from_thread(
                output_md.update,
                f"*Running {template.description}...*"
            )
            
            chunks = chunk_paper(paper, mode="sections")
            content_parts = []
            total_tokens = 0
            max_tokens = 8000
            
            for chunk in chunks:
                if total_tokens + chunk.token_count > max_tokens:
                    break
                content_parts.append(chunk.text)
                total_tokens += chunk.token_count
            
            content = "\n\n---\n\n".join(content_parts)
            system_prompt, user_prompt = template.format(context=content)
            
            generator = client.generate(
                prompt=user_prompt,
                system=system_prompt,
                stream=True,
            )
            
            full_response = ""
            for chunk in generator:
                full_response += chunk
                self.call_from_thread(output_md.update, full_response)
            
            self.call_from_thread(
                self.notify,
                "Analysis complete!",
                severity="information"
            )
    
    @on(Button.Pressed, "#refresh-lib")
    def refresh_library(self) -> None:
        """Refresh the library list."""
        self._load_library()
    
    def on_mount(self) -> None:
        """Called when app is mounted."""
        self._load_library()
    
    @work(thread=True)
    def _load_library(self) -> None:
        """Load library list."""
        from scalpel.embeddings import list_papers
        
        papers = list_papers()
        self.library_papers = papers
        
        def update_list():
            library_list = self.query_one("#library-list", ListView)
            library_list.clear()
            
            if not papers:
                library_list.append(ListItem(Label("No papers in library")))
                return
            
            for paper in papers:
                title = paper["title"]
                if len(title) > 40:
                    title = title[:37] + "..."
                library_list.append(
                    ListItem(Label(f"📄 {title} ({paper['chunk_count']} chunks)"))
                )
        
        self.call_from_thread(update_list)
    
    @on(Button.Pressed, "#remove-paper")
    def remove_paper(self) -> None:
        """Remove selected paper from library."""
        library_list = self.query_one("#library-list", ListView)
        if library_list.index is None or not self.library_papers:
            self.notify("Please select a paper to remove", severity="warning")
            return
        
        if library_list.index >= len(self.library_papers):
            self.notify("Invalid selection", severity="warning")
            return
        
        paper_title = self.library_papers[library_list.index]["title"]
        self._do_remove_paper(paper_title)
    
    @work(thread=True)
    def _do_remove_paper(self, paper_title: str) -> None:
        """Remove paper in background."""
        from scalpel.embeddings import get_store
        
        self.call_from_thread(self.notify, f"Removing: {paper_title[:30]}...")
        
        try:
            store = get_store()
            count = store.delete_paper(paper_title=paper_title)
            self.call_from_thread(
                self.notify, f"Removed {count} chunks", severity="information"
            )
            self.call_from_thread(self._load_library)
        except Exception as e:
            self.call_from_thread(
                self.notify, f"Error removing: {e}", severity="error"
            )
    
    @on(Button.Pressed, "#search-btn")
    def do_search(self) -> None:
        """Run search."""
        search_input = self.query_one("#search-input", Input)
        query = search_input.value.strip()
        
        if not query:
            self.notify("Please enter a search query", severity="warning")
            return
        
        self._run_search(query)
    
    @work
    async def _run_search(self, query: str) -> None:
        """Run search in background."""
        from scalpel.embeddings import search
        
        search_md = self.query_one("#search-md", Markdown)
        search_md.update("*Searching...*")
        
        results = search(query, n_results=5)
        
        if not results:
            search_md.update("No results found.")
            return
        
        md = f"# Search Results for: {query}\n\n"
        
        for i, result in enumerate(results, 1):
            section = f" [{result.section}]" if result.section else ""
            md += f"## {i}. {result.paper_title}{section}\n"
            md += f"**Score:** {result.score:.3f}\n\n"
            md += f"{result.text[:500]}...\n\n---\n\n"
        
        search_md.update(md)


def main():
    """Run the SCALPEL TUI."""
    app = ScalpelApp()
    app.run()


if __name__ == "__main__":
    main()
