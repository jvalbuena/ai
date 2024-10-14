# HelperCrew

HelperCrew is a Python script designed to automate comprehensive research tasks using AI agents. It leverages the capabilities of the Serper API for search and the OpenAI language model to generate insightful content.

## Features

- **Automated Research**: Conducts in-depth analysis on specified topics.
- **Content Generation**: Creates engaging blog posts based on research findings.
- **Customizable Agents**: Define roles and goals for different agents to tailor the research process.

## Requirements

- Python 3.x
- [Serper API](https://serper.dev) - for search capabilities
- [OpenAI API](https://openai.com/api/) - for language model functionalities
- Required Python packages:
  - `crewai`
  - `crewai_tools`
  - `langchain_openai`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HelperCrew.git
   cd HelperCrew
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your API keys as environment variables:
   ```bash
   export SERPER_API_KEY="your_serper_api_key"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

## Usage

Run the script from the command line, providing the research topic and question as arguments:

