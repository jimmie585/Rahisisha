# rahisisha_telegram.py
import os
import logging
import subprocess
import pickle
from pathlib import Path
from telegram import Update  # type: ignore
from telegram.constants import ParseMode  # type: ignore
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters  # type: ignore

from PyPDF2 import PdfReader
from docx import Document as DocxReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- Logging ----------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------- Config ----------------
# Use environment variable for security
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8239679112:AAFVdc9LzKMmpDUniqv-N2WjLyb9MGBB7Yg")
UPLOADS_DIR = Path("uploads")

# ---------------- Document Processing ----------------
def load_document(file_path: str) -> str:
    """Load and extract text from PDF or DOCX files."""
    text = ""
    try:
        if file_path.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        elif file_path.lower().endswith(".docx"):
            doc = DocxReader(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text])
    except Exception as e:
        logger.error(f"❌ Error loading document {file_path}: {e}")
    return text

def split_text(text: str):
    """Split text into chunks for vector storage."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.create_documents([text]) if text else []

# ---------------- Vector Store Handling (Optimized) ----------------
def build_or_update_store(company_dir: Path):
    """Load existing FAISS index and update with new documents incrementally."""
    store_path = company_dir / "vector_store"
    metadata_path = company_dir / "vector_store_metadata.pkl"
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"❌ Failed to load embeddings model: {e}")
        return None

    processed_files = set()
    if metadata_path.exists():
        try:
            with open(metadata_path, "rb") as f:
                processed_files = pickle.load(f)
        except Exception as e:
            logger.warning(f"⚠ Failed to load metadata: {e}")

    vectordb = None
    if store_path.exists() and any(store_path.iterdir()):
        try:
            vectordb = FAISS.load_local(str(store_path), embeddings, allow_dangerous_deserialization=True)
            logger.info(f"✅ Loaded existing vector store for {company_dir.name}")
        except Exception as e:
            logger.warning(f"⚠ Failed to load existing FAISS store: {e}")

    # Process new files
    new_docs = []
    current_files = set()
    
    # Include description.txt
    desc_path = company_dir / "description.txt"
    if desc_path.exists():
        current_files.add("description.txt")
        if "description.txt" not in processed_files:
            desc_text = desc_path.read_text(encoding="utf-8").strip()
            if desc_text:
                desc_chunks = split_text("Company Description:\n\n" + desc_text)
                new_docs.extend(desc_chunks)
                processed_files.add("description.txt")

    # Process document files
    for file in company_dir.glob("*"):
        if file.is_file() and file.suffix.lower() in [".pdf", ".docx"]:
            current_files.add(file.name)
            if file.name not in processed_files:
                text = load_document(str(file))
                if text:
                    chunks = split_text(text)
                    new_docs.extend(chunks)
                    processed_files.add(file.name)

    # Remove files that no longer exist
    processed_files &= current_files

    if new_docs:
        try:
            if vectordb:
                vectordb.add_documents(new_docs)
                logger.info(f"🔹 Added {len(new_docs)} new documents to FAISS store")
            else:
                vectordb = FAISS.from_documents(new_docs, embeddings)
                logger.info(f"✅ Built new FAISS store with {len(new_docs)} documents")
            
            vectordb.save_local(str(store_path))
            with open(metadata_path, "wb") as f:
                pickle.dump(processed_files, f)
        except Exception as e:
            logger.error(f"❌ Failed to save vector store: {e}")
            return None
    elif vectordb is None:
        return None

    return vectordb

# ---------------- AI Query ----------------
def ask_model(context: str, question: str) -> str:
    """Query the Ollama Gemma model with context and question."""
    prompt = f"""You are Raha 🌟, the Rahisisha Virtual Assistant.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, respond exactly with: "❓ I don't know based on the uploaded documents."

Context:
{context}

Question:
{question}

Answer concisely:"""

    try:
        result = subprocess.run(
            ["ollama", "run", "gemma:2b"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )
        
        output = (result.stdout or "").strip()
        
        # Clean up the response
        if "Answer:" in output:
            output = output.split("Answer:", 1)[-1].strip()
        
        # Remove question echo
        output = "\n".join(
            line for line in output.splitlines()
            if not line.strip().lower().startswith("question:")
        ).strip()
        
        if not output or output.lower().strip() in {"i don't know", "idk", "i do not know"}:
            return "❓ I don't know based on the uploaded documents."
            
        return output

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        logger.error(f"❌ Ollama error: {error_msg}")
        return f"❌ Error: {error_msg}"
    except FileNotFoundError:
        logger.error("❌ Ollama not found. Please install Ollama first.")
        return "❌ Ollama is not installed. Please contact support."
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return f"❌ Error: {str(e)}"

# ---------------- Auto Company Loader ----------------
def get_latest_company_dir(user_name: str) -> Path:
    """Find the most recently modified company folder for the user."""
    user_dir = UPLOADS_DIR / user_name
    if not user_dir.exists():
        return None
    
    companies = [d for d in user_dir.iterdir() if d.is_dir()]
    if not companies:
        return None
    
    # Get the company with the most recent activity
    latest_company = max(companies, key=lambda d: d.stat().st_mtime)
    return latest_company

# ---------------- Telegram Commands ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    welcome_msg = (
        "🌟 *Welcome to Rahisisha Virtual Assistant!* 🌟\n\n"
        "I'm *Raha*, your AI-powered company assistant\\.\n"
        "Just ask a question and I'll answer based on your uploaded company documents 📄\n\n"
        "📋 *How to use:*\n"
        "1\\. Upload documents via the Rahisisha dashboard\n"
        "2\\. Ask me anything about your company data\n"
        "3\\. Get instant AI\\-powered answers\\!\n\n"
        "Let's make your business data speak for you\\! 💡"
    )
    
    await update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN_V2)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_msg = (
        "🆘 *Raha Help Guide* 🆘\n\n"
        "📝 *Commands:*\n"
        "• `/start` \\- Welcome message\n"
        "• `/help` \\- Show this help\n"
        "• `/status` \\- Check your company data status\n\n"
        "💬 *Usage:*\n"
        "Simply type any question about your uploaded documents\\!\n\n"
        "📄 *Examples:*\n"
        "• \"What is our company mission?\"\n"
        "• \"List our key products\"\n"
        "• \"What are our compliance policies?\"\n"
        "• \"Summarize the quarterly report\"\n\n"
        "🔧 *Need to upload documents?*\n"
        "Use the Rahisisha web dashboard first\\!"
    )
    
    await update.message.reply_text(help_msg, parse_mode=ParseMode.MARKDOWN_V2)

async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug command to show user info and folder structure."""
    username = update.effective_user.username or f"user_{update.effective_user.id}"
    user_id = update.effective_user.id
    
    debug_msg = f"🔍 **Debug Info:**\n\n"
    debug_msg += f"👤 Telegram Username: {username}\n"
    debug_msg += f"🆔 User ID: {user_id}\n"
    debug_msg += f"📁 Looking in: uploads/{username}/\n\n"
    
    user_dir = UPLOADS_DIR / username
    if user_dir.exists():
        companies = list(user_dir.iterdir())
        debug_msg += f"📂 Found {len(companies)} company folders:\n"
        for company in companies:
            if company.is_dir():
                files = list(company.glob("*"))
                debug_msg += f"  • {company.name}: {len(files)} files\n"
    else:
        debug_msg += f"❌ Directory uploads/{username}/ does not exist\n"
    
    await update.message.reply_text(debug_msg, parse_mode=ParseMode.MARKDOWN)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command to show user's data status."""
    username = update.effective_user.username or f"user_{update.effective_user.id}"
    company_dir = get_latest_company_dir(username)
    
    if not company_dir:
        await update.message.reply_text(
            "❌ *No company data found*\n\n"
            "Please upload documents via the Rahisisha dashboard first\\.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return
    
    company_name = company_dir.name.replace("_", " ")
    doc_files = list(company_dir.glob("*.pdf")) + list(company_dir.glob("*.docx"))
    desc_exists = (company_dir / "description.txt").exists()
    vector_exists = (company_dir / "vector_store").exists()
    
    status_msg = (
        f"✅ *Company Data Status*\n\n"
        f"🏢 *Company:* {company_name}\n"
        f"📄 *Documents:* {len(doc_files)} files\n"
        f"📝 *Description:* {'✅ Available' if desc_exists else '❌ Missing'}\n"
        f"🧠 *AI Index:* {'✅ Ready' if vector_exists else '⚠️ Will build on first query'}\n\n"
        f"You can now ask me questions about your company data\\!"
    )
    
    await update.message.reply_text(status_msg, parse_mode=ParseMode.MARKDOWN_V2)

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user questions about their company documents."""
    username = update.effective_user.username or f"user_{update.effective_user.id}"
    question = update.message.text.strip()
    
    if not question:
        await update.message.reply_text("❓ Please ask a specific question about your company documents.")
        return
    
    company_dir = get_latest_company_dir(username)
    
    if not company_dir:
        await update.message.reply_text(
            "⚠️ *No company data found*\\.\n\n"
            "Please upload documents via the Rahisisha dashboard first\\.\n"
            "Then return here to ask questions\\!",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    # Send "thinking" message
    thinking_msg = await update.message.reply_text("🤔 Raha is analyzing your documents...")
    
    try:
        company_name = company_dir.name.replace("_", " ")
        description_path = company_dir / "description.txt"
        description_text = ""
        
        if description_path.exists():
            try:
                description_text = description_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"Failed to read description: {e}")

        # Build/update vector store
        vectordb = build_or_update_store(company_dir)
        if not vectordb:
            await thinking_msg.edit_text(
                "❌ *No readable documents found*\\.\n\n"
                "Please upload PDF or DOCX files via the dashboard\\.",
                parse_mode=ParseMode.MARKDOWN_V2
            )
            return

        # Retrieve relevant documents
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        results = retriever.get_relevant_documents(question)
        
        if not results:
            await thinking_msg.edit_text(
                "❓ I couldn't find relevant information in your documents to answer this question."
            )
            return
            
        context_docs = "\n\n---\n\n".join([d.page_content for d in results])
        full_context = f"Company Description:\n{description_text}\n\nDocuments:\n{context_docs}"
        
        # Get AI answer
        answer = ask_model(full_context, question)
        
        # Format response
        response_text = (
            f"💡 **Raha's Answer:**\n\n"
            f"{answer}\n\n"
            f"📌 *Based on {company_name}'s uploaded documents*"
        )
        
        await thinking_msg.edit_text(response_text, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        await thinking_msg.edit_text(
            "❌ An error occurred while processing your question. Please try again."
        )

# ---------------- Error Handler ----------------
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors in the bot."""
    logger.error(f"Update {update} caused error {context.error}")
    
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Sorry, something went wrong. Please try again."
        )

# ---------------- Main ----------------
def main():
    """Start the Telegram bot."""
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN environment variable not set!")
        return
    
    # Ensure uploads directory exists
    UPLOADS_DIR.mkdir(exist_ok=True)
    
    # Build application
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("debug", debug_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    
    # Add error handler
    app.add_error_handler(error_handler)

    logger.info("🚀 Raha Telegram bot is running...")
    logger.info(f"📁 Uploads directory: {UPLOADS_DIR.absolute()}")
    
    app.run_polling()

if __name__ == "__main__":
    main()