<p align="center">
  <img src="./assets/AnalystAI-Agent.png" alt="AI Data Analyst Agent banner" width="900" />
</p>

<h1 align="center">📊 AI Data Analyst Agent 🤖</h1>

<p align="center">
  A beginner-friendly Python project that uses <strong>LangChain</strong> and <strong>pandas</strong><br>
  to answer natural-language questions about CSV data from the terminal.
</p>

<p align="center">
  <em>Version 1 (V1) · Readable, tool-based, and intentionally safe.</em>
</p>

## Overview

AI Data Analyst Agent is a lightweight applied AI project that enables natural-language analysis of CSV datasets from the terminal. Built with LangChain and pandas, it demonstrates how an LLM-powered agent can use structured tools to answer data questions, surface insights, and support common exploratory analysis tasks.

Instead of relying on unrestricted Python execution or a hidden REPL, the project uses a defined set of custom analysis tools for summaries, missing-value checks, grouped aggregations, rankings, and basic correlation analysis. This approach keeps the system safer, more interpretable, and easier to maintain.

## Core Capabilities

| Capability | Description |
|---|---|
| Natural-language querying | Ask questions about CSV data directly from the terminal in plain English. |
| Dataset inspection | Review columns, structure, previews, and summary statistics quickly. |
| Data quality checks | Identify missing values and inspect incomplete fields. |
| Grouped analysis | Compare categories using grouped aggregations such as averages, sums, and counts. |
| Ranking and filtering | Surface top-performing rows or categories based on selected metrics. |
| Safe execution model | Uses predefined analysis tools instead of unrestricted Python execution. 

## Usage

The app follows a simple flow:

1. Load a CSV file into a pandas DataFrame
2. Wrap a small set of safe analysis functions as LangChain tools
3. Give those tools to a LangChain agent
4. Let the user ask questions in a terminal loop
5. Return answers in plain English

The agent does **not** execute arbitrary Python code.  
It can only use the tools defined in `tools.py`.

### 📌 Example
```text
You: Give me a summary of the data

Agent:
The dataset contains 20 rows and 10 columns. The columns include both
categorical and numeric data. The categorical columns are product_id,
product_name, category, region, and month. The numeric columns are
units_sold, unit_price, revenue, ad_spend, and sales. If you need more
specific details or analysis, please let me know.

You: Are there missing values?

Agent:
Yes, there are missing values in the dataset. Specifically, the columns
"ad_spend" and "sales" each have 1 missing value. All other columns have
no missing values.
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
