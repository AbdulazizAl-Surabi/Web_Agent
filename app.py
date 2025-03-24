import os
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Union

# Third-party imports
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Local module imports
from browser_use import Agent

load_dotenv()


@dataclass
class ActionResult:
    is_done: bool
    extracted_content: Optional[str]
    error: Optional[str]
    include_in_memory: bool


@dataclass
class AgentHistoryList:
    all_results: List[ActionResult]
    all_model_outputs: List[dict]


def parse_agent_history(history_str: str) -> None:
    console = Console()
    sections = history_str.split('ActionResult(')
    for i, section in enumerate(sections[1:], 1):  # Überspringe den ersten leeren Abschnitt
        content = ''
        if 'extracted_content=' in section:
            content = section.split('extracted_content=')[1].split(',')[0].strip("'")
        if content:
            header = Text(f'Schritt {i}', style='bold blue')
            panel = Panel(content, title=header, border_style='blue')
            console.print(panel)
            console.print()


async def process_agent_task(
    task: str,
    api_key: str,
    model: str = 'gpt-4',
    headless: bool = True,
    pdf_file: Optional[Union[dict, str]] = None,
) -> str:
    if not api_key.strip():
        return "Bitte einen API-Key angeben."

    os.environ['OPENAI_API_KEY'] = api_key

    pdf_info = "Kein PDF hochgeladen."
    if pdf_file is not None:
        # Prüfen, ob pdf_file ein Dictionary ist oder direkt ein Dateipfad (String)
        if isinstance(pdf_file, dict):
            pdf_info = f"PDF hochgeladen: {pdf_file.get('name', 'unbekannt')}"
        else:
            pdf_info = f"PDF hochgeladen: {pdf_file}"

    try:
        agent = Agent(
            task=task,
            llm=ChatOpenAI(model=model),
        )
        result = await agent.run()
        return f"{pdf_info}\n\nAgent Result:\n{result}"
    except Exception as e:
        return f"Fehler beim Agent-Task: {e}"


def extract_pdf(pdf_file: Optional[Union[dict, str]], output_method: str) -> str:
    if not pdf_file:
        return "Bitte ein PDF-Dokument hochladen."

    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        # Bestimme den Dateipfad: entweder aus dem Dictionary oder direkt als String
        if isinstance(pdf_file, dict):
            pdf_path = pdf_file.get("name")
        else:
            pdf_path = pdf_file

        if not pdf_path:
            return "Fehler: PDF-Dateipfad nicht gefunden."

        result = converter.convert(pdf_path)
        markdown_output = result.document.export_to_markdown()
        if output_method == "Lokal speichern":
            output_filename = "converted_output.md"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(markdown_output)
            return f"PDF wurde konvertiert und in '{output_filename}' gespeichert."
        else:
            return markdown_output
    except Exception as e:
        return f"Fehler bei der PDF-Konvertierung: {e}"


def create_ui():
    with gr.Blocks(title="DATA ACQUISITION") as interface:
        gr.Markdown("# Finde hochwertige Inhalte online")

        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
                task = gr.Textbox(label="Topic", placeholder='z.B. "neueste AI Nachrichten"', lines=3)
                model = gr.Dropdown(choices=["gpt-4", "gpt-3.5-turbo"], label="Model", value="gpt-4")
                headless = gr.Checkbox(label="Run Headless", value=True)
                pdf_file = gr.File(label="PDF Datei", file_count="single", file_types=[".pdf"])
                submit_btn = gr.Button("Agent-Task ausführen")
            with gr.Column():
                output_agent = gr.Textbox(label="Agent Output", lines=15, interactive=False)

        with gr.Row():
            with gr.Column():
                output_method = gr.Radio(
                    choices=["Gradio anzeigen", "Lokal speichern"],
                    label="Ausgabemethode",
                    value="Gradio anzeigen"
                )
                extract_btn = gr.Button("Extract PDF")
            with gr.Column():
                output_pdf = gr.Textbox(label="PDF Output", lines=15, interactive=False)

        submit_btn.click(
            fn=lambda task, api_key, model, headless, pdf_file: asyncio.run(
                process_agent_task(task, api_key, model, headless, pdf_file)
            ),
            inputs=[task, api_key, model, headless, pdf_file],
            outputs=output_agent,
        )

        extract_btn.click(
            fn=extract_pdf,
            inputs=[pdf_file, output_method],
            outputs=output_pdf,
        )

    return interface


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
