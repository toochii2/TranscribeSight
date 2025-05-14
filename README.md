# TranscribeSight

**Your Intelligent Speech-to-Text Evaluation Platform**

A comprehensive tool for comparing and analysing various speech-to-text transcription services including AssemblyAI, OpenAI Whisper, GPT-4o, Deepgram, and Google Gemini models.

![TranscribeSight Dashboard](Images\1.png)

## Features

- **Multi-Service Comparison**: Compare transcription from 8 different STT services:
  - AssemblyAI
  - OpenAI Whisper
  - OpenAI GPT-4o
  - OpenAI GPT-4o Mini
  - Deepgram Nova 3
  - Google Gemini Pro 2.5
  - Google Gemini 2.0 Flash
  - Google Gemini 2.5 Flash

- **Comprehensive Metrics**:
  - Word Error Rate (WER)
  - Character Error Rate (CER)
  - Token Error Rate (TER)
  - Match Error Rate (MER)
  - Word Information Preserved/Lost
  - Processing time
  - Cost analysis
  
- **Semantic Analysis** (optional):
  - SeMaScore (semantic similarity)
  - STS (Semantic Textual Similarity)
  - BERTScore
  - LLM-based Meaning Preservation
  - SWWER (Semantic-Weighted Word Error Rate)

- **Visualisations**:
  - Performance metrics charts
  - Accuracy analysis
  - Radar charts for multi-dimensional comparison
  - Cost-efficiency analysis

- **AI-Powered Analysis**:
  - Generate comprehensive reports about transcription performance
  - Service strengths and weaknesses
  - Targeted recommendations based on use cases

- **Export Options**:
  - Download results as Excel spreadsheets
  - Export visualisations as PNG

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/TranscribeSight.git
cd TranscribeSight
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Requirements

- Python 3.8+
- See requirements.txt for all Python dependencies

## API Keys

You'll need API keys for the services you want to compare:
- [AssemblyAI](https://www.assemblyai.com/)
- [OpenAI](https://platform.openai.com/)
- [Deepgram](https://deepgram.com/)
- [Google AI (Gemini)](https://ai.google.dev/)

## Usage

1. Enter your API keys in the sidebar
2. Upload audio files (.wav format)
3. Upload corresponding reference text files (optional, for accuracy analysis)
4. Select which models to include in the comparison
5. Choose semantic analysis options (if needed)
6. Click "Start Processing"
7. Navigate through the tabs to view different analyses
8. Download results as Excel spreadsheets or export visualisations

## License

TranscribeSight is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the web application framework
- All the STT service providers for their APIs