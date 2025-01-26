https://au-tax-bot.streamlit.app

# Australian Tax Legislation Analysis Tool 🔍⚖️

A sophisticated Python application that leverages AI to analyze Australian tax legislation, providing comprehensive insights and interpretations through an interactive web interface.

## 🌟 Key Features

- **AI-Powered Analysis**: Utilizes OpenAI's GPT models for intelligent interpretation
- **Smart Query Processing**: Enhanced query understanding with tax-specific context
- **Legislative Search**: Advanced similarity-based search using embeddings
- **Interactive UI**: User-friendly Streamlit interface
- **Source References**: Direct links to original legislation
- **Configurable Settings**: Adjustable parameters for precise results

## 🛠️ Technical Stack

- Python 3.x
- OpenAI API (GPT-3.5/GPT-4)
- Streamlit
- Pandas
- TikToken
- SciPy

## 📋 Prerequisites

Before running the application, ensure you have:

- Python 3.x installed
- OpenAI API key
- `tax_embeddings.json` file
- Required Python packages

## 🔧 Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd tax-legislation-analysis

2. Install required packages:
   ```bash
   pip install -r requirements.txt

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'

## 📦 Required Packages

- `openai`
- `pandas`
- `tiktoken`
- `scipy`
- `streamlit`

---

## 🚀 Usage

1. Start the application:
   ```bash
   streamlit run app.py
2. Access the web interface at [http://localhost:8501](http://localhost:8501).
3. Enter your tax-related query in the text area.
4. Configure advanced options if needed:
   - Number of relevant sections.
   - Minimum similarity score.
   - GPT model selection.
5. Click **"Analyze Legislation"** to get results.

---

## 💡 Features in Detail

### Query Processing

- Preprocesses and enhances user queries.
- Adds Australian tax context.
- Standardizes tax terminology.

### Legislative Analysis

- Retrieves relevant sections using embedding similarity.
- Filters results based on configurable thresholds.
- Generates comprehensive analysis using GPT models.

### User Interface

- Intuitive query input.
- Advanced configuration options.
- Expandable legislative references.
- Direct links to source legislation.

---

## ⚠️ Error Handling

- Robust error handling for API issues.
- Graceful handling of data loading failures.
- User-friendly error messages.
- Detailed logging for troubleshooting.

---

## 🔒 Security

- Secure API key handling.
- Environment variable support.
- Password-protected API key input.

---

## 📝 Logging

- Comprehensive logging system.
- Error tracking and reporting.
- Performance monitoring.

---

## ⚖️ Disclaimer

This tool provides information based on tax legislation but should not be considered as legal or tax advice. Always consult with a qualified tax professional for specific advice about your situation.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
