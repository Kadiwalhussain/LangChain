# Quick Start Guide - Financial Assistant

## 🚀 Getting Started

### Step 1: Installation

```bash
cd FinancialAssistant
pip install -r requirements.txt
```

### Step 2: Configuration

Copy the environment template:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```bash
# For OpenAI
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=openai

# OR for Ollama (local, free)
LLM_PROVIDER=ollama
OLLAMA_MODEL=mistral
```

### Step 3: Start Ollama (if using local models)

```bash
# Install Ollama from https://ollama.ai
ollama pull mistral
ollama pull nomic-embed-text
```

### Step 4: Index Your Documents

```bash
# CLI
python cli.py index-documents --directory ./data/knowledge_base

# Or interactively
python cli.py index-documents
```

## 💡 Usage Examples

### Command Line Interface

#### Ask Questions
```bash
python cli.py ask "What home office deductions can I claim?"
```

#### Interactive Mode
```bash
python cli.py interactive
```

#### Analyze a Form
```bash
python cli.py analyze-form ./data/tax_forms/W2_2024.pdf
```

#### Find Deductions
```bash
python cli.py find-deductions
```

### API Server

#### Start the Server
```bash
python app.py
```

Server runs at: `http://localhost:8000`

#### API Documentation
Visit: `http://localhost:8000/docs`

#### Example API Calls

**Ask a Question:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are qualified business expenses?"}'
```

**Upload a Document:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@tax_form.pdf"
```

**Find Deductions:**
```bash
curl -X POST "http://localhost:8000/deductions" \
  -H "Content-Type: application/json" \
  -d '{
    "profession": "Software Engineer",
    "expenses": ["home_office", "computer", "internet"],
    "income": 120000,
    "filing_status": "single"
  }'
```

## 🎯 Common Use Cases

### 1. Tax Preparation Helper

```python
from src.qa_chain import FinancialQAChain

qa = FinancialQAChain()

# Get deduction information
result = qa.ask("What documents do I need for home office deduction?")
print(result['answer'])
```

### 2. Form Validation

```python
from src.form_analyzer import FormAnalyzer

analyzer = FormAnalyzer()
result = analyzer.analyze_form("path/to/form.pdf")

print(f"Completeness: {result['completeness_score']}%")
print(f"Missing: {result['missing_fields']}")
```

### 3. Deduction Discovery

```python
from src.deduction_finder import DeductionFinder

finder = DeductionFinder()
deductions = finder.find_deductions(
    profession="Freelancer",
    expenses=["home_office", "software", "internet"],
    income=80000
)

for d in deductions:
    print(f"{d['name']}: ${d['estimated_savings']}")
```

## 📊 Understanding Results

### QA Response Format
```json
{
  "question": "Your question",
  "answer": "AI generated answer with context",
  "sources": [
    {
      "file": "document_name.pdf",
      "category": "tax_form",
      "page": 2
    }
  ]
}
```

### Form Analysis Format
```json
{
  "form_type": "W-2",
  "completeness_score": 85.5,
  "missing_fields": ["box_12_code", "state_wages"],
  "warnings": ["Potential issue with..."],
  "recommendations": ["Consider adding..."],
  "summary": "Overall assessment"
}
```

## 🛠️ Troubleshooting

### Issue: "No documents found"
**Solution:** Index documents first:
```bash
python cli.py index-documents
```

### Issue: "Connection refused" (Ollama)
**Solution:** Start Ollama:
```bash
ollama serve
```

### Issue: "Out of memory"
**Solution:** Reduce chunk size in config.py:
```python
chunk_size = 500  # Down from 1000
```

### Issue: "Slow responses"
**Solution:** 
- Use local Ollama instead of OpenAI
- Reduce top_k_results in config
- Use faster embedding model

## 📝 Tips & Best Practices

1. **Organize Documents by Year**
   ```
   data/user_uploads/
   ├── 2023/
   │   ├── W2_2023.pdf
   │   └── 1099_2023.pdf
   └── 2024/
       └── receipts/
   ```

2. **Use Descriptive Filenames**
   - Good: `W2_Company_2024.pdf`
   - Bad: `document1.pdf`

3. **Keep Backups**
   - Vector store is in `./database/chroma_db/`
   - Back this up regularly

4. **Update Knowledge Base**
   - Add new tax law changes
   - Re-index after adding files

## 🔐 Security Notes

- Never commit `.env` file
- Don't share API keys
- Keep sensitive documents local
- Use encryption for production

## 📚 Next Steps

- Explore the API at `/docs`
- Add your own tax documents
- Customize prompts in `src/qa_chain.py`
- Add more knowledge base files
- Build a frontend interface

## 🤝 Need Help?

- Check logs in console output
- Review error messages carefully
- Consult LangChain documentation
- Ensure all dependencies are installed

---

**Happy Filing! 💰**
