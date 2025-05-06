from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from googletrans import Translator


# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

translator = Translator()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "YOUR_GOOGLE_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)

# Set up the model
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# Initialize Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def extract_images_from_pdf(pdf_bytes):
    """Extract images from PDF pages."""
    images = []
    try:
        # Open the PDF from bytes
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Process each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Get page as an image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_bytes = pix.tobytes("png")
            
            images.append({
                "page": page_num + 1,
                "mime_type": "image/png",
                "data": img_bytes
            })
        
        return images
    except Exception as e:
        raise Exception(f"Error extracting images from PDF: {str(e)}")
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

def analyze_handwritten_content(image_data, page_num):
    """
    Analyze handwritten content in an image using Gemini.
    
    Args:
        image_data (bytes): The image data
        page_num (int): Page number for reference
        
    Returns:
        dict: Extracted text and analysis
    """
    try:
        # System prompt for handwritten text extraction
        system_prompt = """
        You are an expert at extracting and transcribing handwritten text from images.
        
        Please follow these instructions:
        1. Carefully examine the image and identify all handwritten text.
        2. Transcribe the handwritten text accurately, preserving the original structure.
        3. If there are multiple sections or paragraphs, preserve that structure.
        4. If some text is unclear or illegible, indicate this with [illegible] or try to extract if it is able to extract with atleast 85% plus accuracy.
        5. Focus  on the text content, any drawings or non-text elements also.
        6. Present the transcription in a clean, readable format.
        
        Respond with just the transcribed text in proper markdown format(you should generate compilable markdown for tables also), without any additional commentary.
        """
        
        # Prepare image parts for Gemini API
        prompt_parts = [
            {
                "mime_type": "image/png",
                "data": image_data
            },
            system_prompt
        ]
        
        # Generate response using Gemini API
        response = model.generate_content(prompt_parts)
        
        extracted_text = response.text if response else ""
        
        return {
            "page": page_num,
            "extracted_text": extracted_text
        }
    
    except Exception as e:
        return {
            "page": page_num,
            "error": str(e),
            "extracted_text": ""
        }

def summarize_content(extracted_texts):
    """
    Generate a summary of the extracted content.
    
    Args:
        extracted_texts (list): List of extracted text from all pages
        
    Returns:
        str: Summary of the content
    """
    if not extracted_texts:
        return "No content was extracted to summarize."
    
    full_text = "\n\n".join([f"Page {item['page']}:\n{item['extracted_text']}" for item in extracted_texts])
    
    system_prompt = """
    You are an expert at summarizing handwritten document content.
    
    Please follow these instructions:
    1. Read through the provided transcription of the handwritten document.
    2. Create a comprehensive summary that captures the key points and main ideas.
    3. Organize the summary in a clear, structured format.
    4. Highlight any important details, dates, names, or action items.
    5. Keep the summary concise yet thorough.
    6. If there are unclear sections marked as [illegible], note that some information may be missing.
    
    Structure your summary with clear headings and bullet points as appropriate.
    """
    
    try:
        response = model.generate_content([
            system_prompt,
            f"Here is the transcribed handwritten content to summarize:\n\n{full_text}"
        ])
        
        return response.text if response else "Unable to generate summary."
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

@app.post("/extract-handwriting/")
async def extract_handwriting(file: UploadFile = File(...)):
    """
    Extract handwritten content from an uploaded PDF.
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
        
        # Read the file
        pdf_bytes = await file.read()
        
        # Extract images from PDF pages
        images = extract_images_from_pdf(pdf_bytes)
        
        if not images:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No pages were found in the PDF."}
            )
        
        # Process each page image to extract handwritten content
        extracted_contents = []
        for image in images:
            result = analyze_handwritten_content(image["data"], image["page"])
            extracted_contents.append(result)
        
        # Generate summary of all extracted content
        summary = summarize_content(extracted_contents)
        
        return JSONResponse(
            content={
                "status": "success",
                "pages": len(images),
                "extracted_contents": extracted_contents,
                "summary": summary
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
    

@app.post("/translate/")
async def translate_text(
    text: str = Form(...),
    target_language: str = Form(...)
):
    """
    Translate given text into target_language using googletrans.
    Accepts UI names like 'hindi', 'tamil', etc.
    """
    # Map your UI's language names → googletrans codes
    lang_map = {
        "hindi":   "hi",
        "tamil":   "ta",
        "telugu":  "te",
        "kannada": "kn",
        "malayalam":"ml",
        "marathi": "mr",
        "bengali": "bn",
        "gujarati":"gu",
        "punjabi": "pa",
        "odia":    "or",
        "english": "en"
    }
    code = lang_map.get(target_language.lower(), target_language)
    try:
        # translator.translate is a coroutine here, so we await it
        result = await translator.translate(text, dest=code)
        return JSONResponse({"translated_text": result.text})
    except Exception as e:
        raise HTTPException(500, str(e))
    

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Handwritten PDF extractor API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)