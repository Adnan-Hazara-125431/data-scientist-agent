# Data Scientist Agent

A multi-agent system for automated data analysis. Upload a messy CSV or Excel file, ask a question, and the agent will clean your data, write pandas analysis code, self-heal errors, and produce charts with a human-readable summary.

## Architecture

```
User Upload + Question
        │
        ▼
  Orchestrator (CrewAI Crew)
        │
        ├── 1. Data Cleaner    → fix dates, fill missing values
        ├── 2. Code Writer     → write pandas analysis code
        ├── 3. Code Healer     → execute & fix errors (up to 5 retries)
        └── 4. Visualizer      → charts + narrative summary
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini (`langchain-google-genai`) |
| Orchestration | CrewAI |
| UI | Streamlit |
| Data Processing | pandas |
| Charts | matplotlib + seaborn |

## Project Structure

```
data-scientist-agent/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── llm.py              # Shared Gemini LLM config
│   │   ├── orchestrator.py     # Pipeline coordinator
│   │   ├── cleaner.py          # Data cleaning agent
│   │   ├── code_writer.py      # Analysis code agent
│   │   ├── code_healer.py      # Self-healing debugger agent
│   │   └── visualizer.py       # Charts & summary agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # CSV/Excel loader
│   │   ├── code_executor.py    # Safe Python executor
│   │   └── chart_generator.py  # Matplotlib chart saver
│   ├── main.py                 # CLI pipeline test
│   └── app.py                  # Streamlit UI
├── .env
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
cd data-scientist-agent
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API key

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env
```

Edit `.env` and set your key:

```
GEMINI_API_KEY=your-actual-api-key
```

## Usage

### Streamlit UI (recommended)

```bash
streamlit run src/app.py
```

1. Upload a CSV or Excel file
2. Enter your analysis question
3. Click **Analyze**
4. View cleaned data, charts, and summary

### CLI pipeline test

Runs the full agent pipeline on mock messy data (no file upload needed):

```bash
python -m src.main
```

## The 5 Agents

| Agent | Role |
|-------|------|
| **Orchestrator** | Manages workflow, delegates to other agents |
| **Data Cleaner** | Standardizes columns, fixes dates, fills missing values |
| **Code Writer** | Writes pandas code to answer the user's question |
| **Code Healer** | Executes code, catches errors, retries up to 5 times |
| **Visualizer** | Creates matplotlib charts and writes a plain-language summary |

## Example Questions

- "What is the total revenue by region?"
- "Which product has the highest average sales?"
- "Show me the trend of units sold over time."
- "Are there any outliers in the revenue column?"
- "Summarize the key patterns in this dataset."

## Output

The pipeline produces:

- **Cleaned DataFrame** — standardized, imputed, deduplicated
- **Analysis output** — printed statistics and computed results
- **Charts** — saved to `output/charts/`
- **Summary** — human-readable narrative answering your question

## License

MIT
