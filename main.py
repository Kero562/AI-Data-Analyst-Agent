import argparse
import os
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from data_loader import get_dataset_shape, load_csv
from prompts import DATA_ANALYST_SYSTEM_PROMPT
from tools import create_dataframe_tools


SAMPLE_CSV_PATH = Path("sample_data") / "sales_data.csv"
EXAMPLE_QUESTIONS = [
    "What columns are in this dataset?",
    "Give me a summary of the data.",
    "Which category has the highest average sales?",
    "Are there any missing values?",
    "Show the top 5 products by revenue.",
    "Is there a correlation between ad_spend and sales?",
]


def print_welcome() -> None:
    """Show a simple startup message."""
    print("=" * 60)
    print("AI Data Analyst Agent")
    print("CSV analysis with LangChain tools and pandas")
    print("=" * 60)


def main() -> None:
    """Run the CLI app."""
    load_dotenv()
    args = parse_args()
    print_welcome()

    csv_path = resolve_csv_path(args.csv)
    dataframe = load_csv(csv_path)
    row_count, column_count = get_dataset_shape(dataframe)
    dataframe_tools = create_dataframe_tools(dataframe)

    print(f"Loaded file: {csv_path}")
    print(f"Rows: {row_count}")
    print(f"Columns: {column_count}")
    print(f"Tools: {len(dataframe_tools)}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nOpenAI API key not found.")
        print("Set OPENAI_API_KEY in your environment or in a local .env file.")
        print("The agent code is ready, but the model cannot run without a key.")
        return

    agent = build_data_analyst_agent(dataframe_tools)

    if args.demo:
        run_demo_questions(agent)
        if args.no_interactive:
            return

    run_cli_loop(agent)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AI Data Analyst Agent")
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a CSV file. If omitted, the app prompts you and falls back to sample data.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a few example questions before starting the interactive CLI.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Exit after demo mode instead of starting the question loop.",
    )
    return parser.parse_args()


def resolve_csv_path(csv_arg: str | None) -> Path:
    """Return the CSV path for this session."""
    if csv_arg:
        return Path(csv_arg)

    user_input = input(
        f"CSV path (press Enter to use {SAMPLE_CSV_PATH.as_posix()}): "
    ).strip()

    if user_input:
        return Path(user_input)

    return SAMPLE_CSV_PATH


def build_data_analyst_agent(dataframe_tools: list[BaseTool]):
    """Create a LangChain agent for the loaded dataset."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    model = ChatOpenAI(model=model_name, temperature=0)

    return create_agent(
        model=model,
        tools=dataframe_tools,
        system_prompt=DATA_ANALYST_SYSTEM_PROMPT,
    )


def run_demo_questions(agent: Any) -> None:
    """Run a few example questions to show how the agent behaves."""
    print("\nDemo questions:")
    for question in EXAMPLE_QUESTIONS[:3]:
        print(f"\n> {question}")
        print(ask_agent(agent, question))


def run_cli_loop(agent: Any) -> None:
    """Start the terminal question loop."""
    print("\nAsk a question about the dataset.")
    print("Type 'examples' to see sample questions or 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if question.lower() == "examples":
            print_examples()
            continue

        try:
            answer = ask_agent(agent, question)
            print(f"\nAgent:\n{answer}")
        except Exception as error:
            print(f"\nAgent error:\n{error}")


def ask_agent(agent: Any, question: str) -> str:
    """Send one question to the agent and return the final text answer."""
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )
    return extract_final_answer(result)


def extract_final_answer(result: dict[str, Any]) -> str:
    """Pull the final text answer out of the agent result."""
    messages = result["messages"]
    final_message = messages[-1]
    content = final_message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_blocks.append(block.get("text", ""))

        if text_blocks:
            return "\n".join(text_blocks)

    return str(content)


def print_examples() -> None:
    """Print example questions for the CLI."""
    print("\nExample questions:")
    for question in EXAMPLE_QUESTIONS:
        print(f"- {question}")


if __name__ == "__main__":
    main()
