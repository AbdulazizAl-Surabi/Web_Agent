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
    for i, section in enumerate(sections[1:], 1):  # Skip the first empty section
        content = ''
        if 'extracted_content=' in section:
            content = section.split('extracted_content=')[1].split(',')[0].strip("'")
        if content:
            header = Text(f'Step {i}', style='bold blue')
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
        return "Please provide an API key."

    os.environ['OPENAI_API_KEY'] = api_key

    pdf_info = "No PDF uploaded."
    if pdf_file is not None:
        if isinstance(pdf_file, dict):
            pdf_info = f"PDF uploaded: {pdf_file.get('name', 'unknown')}"
        else:
            pdf_info = f"PDF uploaded: {pdf_file}"

    try:
        agent = Agent(
            task=task,
            llm=ChatOpenAI(model=model),
        )
        result = await agent.run()
        return f"{pdf_info}\n\nAgent Result:\n{result}"
    except Exception as e:
        return f"Error during agent task: {e}"


def extract_pdf(pdf_file: Optional[Union[dict, str]], output_method: str) -> str:
    if not pdf_file:
        return "Please upload a PDF document."

    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        if isinstance(pdf_file, dict):
            pdf_path = pdf_file.get("name")
        else:
            pdf_path = pdf_file

        if not pdf_path:
            return "Error: PDF file path not found."

        result = converter.convert(pdf_path)
        markdown_output = result.document.export_to_markdown()
        if output_method == "Save Locally":
            output_filename = "converted_output.md"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(markdown_output)
            return f"PDF converted and saved as '{output_filename}'."
        else:
            return markdown_output
    except Exception as e:
        return f"Error during PDF conversion: {e}"


def create_ui():
    with gr.Blocks(title="DATA ACQUISITION") as interface:
        # Centered title with BLUESCOUT and "BLUE" in caps and colored #0E5FF6
        gr.Markdown("<h1 style='text-align: center; font-size: 60px;'><span style='color: #0E5FF6;'>BLUE</span>Scout</h1>")
        gr.Markdown("## Find High-Quality Content Online")

        with gr.Row():
            # Left Column: LLM Agent
            with gr.Column(scale=1):
                gr.Markdown("### LLM Agent Prompt")
                api_key = gr.Textbox(label="OpenAI API Key", placeholder="sk-...", type="password")
                task = gr.Textbox(label="Topic", placeholder='e.g. "latest AI news"', lines=3)
                model = gr.Dropdown(choices=["gpt-4", "gpt-3.5-turbo"], label="Model", value="gpt-4")
                headless = gr.Checkbox(label="Run Headless", value=True)
                submit_btn = gr.Button("Execute Agent Task")
                output_agent = gr.Textbox(label="Agent Output", lines=15, interactive=False)

            # Right Column: PDF Extraction
            with gr.Column(scale=1):
                gr.Markdown("### PDF Extraction")
                pdf_file = gr.File(label="PDF File (Drag & Drop)", file_count="single", file_types=[".pdf"])
                output_method = gr.Radio(
                    choices=["Display in Gradio", "Save Locally"],
                    label="Output Method",
                    value="Display in Gradio"
                )
                extract_btn = gr.Button("Extract PDF")
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
