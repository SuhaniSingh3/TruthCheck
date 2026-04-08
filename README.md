# TruthCheck - AI-Powered Fake News Detection

TruthCheck is a modern, high-performance web application designed to detect and analyze news authenticity. It leverages **Groq's LPU™ Inference Engine** and **Meta's Llama 3.3 (70B)** to provide ultra-fast, reasoning-based analysis of news articles.

## ✨ Features
- **Ultra-Fast Analysis**: Powered by Groq for near-instantaneous predictions.
- **Explainable AI**: Provides detailed reasoning and summaries for every verdict (REAL vs FAKE).
- **Premium Glassmorphism UI**: Beautiful, responsive, and modern interface with dynamic themes.
- **RESTful API**: Simple endpoints for integration with other services.

## 🛠️ Tech Stack
- **Backend**: Flask (Python)
- **AI Engine**: Groq Cloud SDK (Llama 3.3 70B Versatile)
- **Frontend**: Glassmorphism CSS / Vanilla JavaScript
- **Deployment**: Vercel Compatible

## 📂 Project Structure
```
fake-news-detection/
├── app.py                # Core Flask application & Groq logic
├── templates/            # Frontend HTML files (index.html, result.html)
├── requirements.txt      # Python dependencies
├── vercel.json           # Vercel deployment configuration
├── .env                  # API Key configuration
└── README.md             # This file
```

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/SuhaniSingh3/TruthCheck.git
cd TruthCheck

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and add your Groq API Key:
```bash
GROQ_API_KEY=gsk_your_key_here
```
*You can get an API key at [console.groq.com](https://console.groq.com/)*

### 3. Run Locally
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

## 🌐 API Usage

### `POST /predict`
Analyze a news article.
**Request:**
```json
{
  "text": "The full content of the news article here..."
}
```

**Response:**
```json
{
  "success": true,
  "label": "FAKE NEWS",
  "prediction": 1,
  "confidence": 98.5,
  "reasons": ["Lacks credible sources", "Emotional triggers detected"],
  "summary": "This article contains several markers of misinformation...",
  "source": "Groq (Llama-3.3)"
}
```

## 📜 License
MIT License - Open for all.

---
*Created with ❤️ for a more truthful internet.*
