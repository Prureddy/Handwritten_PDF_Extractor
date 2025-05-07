# Handwritten PDF Extractor

A web-based tool that leverages AI to convert scanned or handwritten PDF documents into structured digital text. Upload your PDFs, get accurate Markdown transcriptions, concise AI-generated summaries, and instant translations into multiple languages.

---

## 🚀 Live Demo

* **Frontend**: [https://handwritten-frontend.onrender.com](https://handwritten-frontend.onrender.com)
* **Backend API**: [https://handwritten-backend.onrender.com/docs](https://handwritten-backend.onrender.com/docs)

---

## 📖 Features

* **Handwritten Text Extraction**: Uses Google’s Gemini model to transcribe handwriting into Markdown.
* **Smart Summaries**: AI-generated summaries organized with headings and bullet points.
* **Multilingual Translations**: Translate summaries into Hindi, Tamil, Telugu, and more via Google Translate.
* **Interactive UI**: Drag & drop PDF upload, progress indicator, copy/download buttons, and tabbed view for summary vs. full transcription.
* **Export Options**: Download summaries and full text as TXT or DOCX.
* **Security**: CORS restricted to authorized origins, environment variables for API keys.

---

## 🏗️ Tech Stack

* **Backend**: FastAPI, Uvicorn, PyMuPDF, google-generativeai, googletrans
* **Frontend**: HTML, CSS (Flexbox), Vanilla JavaScript, [marked.js](https://github.com/markedjs/marked)
* **AI Services**: Google Generative Language API (Gemini), Google Translate API
* **Deployment**: Render.com (Web Service + Static Site)

---

## 🎯 Getting Started

### Prerequisites

* Python 3.8+
* Node.js/npm (optional, for local static server)

### Backend Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/handwritten-extractor.git
cd handwritten-extractor/backend

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Create a .env file
echo "GOOGLE_API_KEY=<YOUR_API_KEY>" > .env
echo "FRONTEND_URL=http://localhost:8080" >> .env

# Run the API server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# Serve static files locally
cd ../frontend
python -m http.server 8080
```

Open your browser at `http://localhost:8080`.

---

## 🔧 Configuration & Environment Variables

| Variable         | Description                                     | Example                                     |
| ---------------- | ----------------------------------------------- | ------------------------------------------- |
| `GOOGLE_API_KEY` | API key for Google Generative Language (Gemini) | `AIzaSy…`                                   |
| `FRONTEND_URL`   | Allowed CORS origin for the frontend            | `https://handwritten-frontend.onrender.com` |

On Render, set these under **Environment** → **Environment Variables** for your backend service.

---

## 🛠️ Usage

1. Upload a handwritten PDF via the drag-and-drop area or browse dialog.
2. Click **Process PDF** and watch the progress bar.
3. View the AI-generated summary under the **Summary** tab.
4. Switch to **Extracted Content** for full transcription.
5. Translate the summary using the language selector and **Translate** button.
6. Copy or download results as TXT or DOCX.

---


---

## 🤝 Contributing

Contributions welcome! Please open issues or pull requests in the [GitHub repo](https://github.com/Prureddy/handwritten-extractor).

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
