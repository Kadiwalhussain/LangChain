"""
YouTube Video Loader and Q&A System
===================================

Loads YouTube video transcripts and enables Q&A using local models.
No API keys required!
"""

import os
import re
from typing import Dict, List, Optional
import logging

try:
    from langchain_community.document_loaders import YoutubeLoader
except ImportError:
    # Fallback if YoutubeLoader not available
    YoutubeLoader = None
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class YouTubeLoader:
    """Load YouTube videos and enable Q&A."""
    
    def __init__(self):
        """Initialize YouTube loader with local models."""
        # Use local embeddings (no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # Free, local model
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Vector stores for different videos
        self.vectorstores = {}
        
        # LLM (local, no API key)
        try:
            self.llm = OllamaLLM(model="mistral", temperature=0.2)
            logger.info("✅ Ollama LLM initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama: {e}")
            logger.error("   Please ensure Ollama is running: ollama serve")
            logger.error("   And pull a model: ollama pull mistral")
            self.llm = None
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def load_video(self, video_id: str) -> Dict:
        """Load YouTube video transcript and create vector store."""
        try:
            logger.info(f"Loading YouTube video: {video_id}")
            
            if YoutubeLoader is None:
                return {
                    'success': False,
                    'error': 'YoutubeLoader not available. Install: pip install youtube-transcript-api'
                }
            
            # Load transcript
            loader = YoutubeLoader.from_youtube_id(
                video_id,
                add_video_info=True
            )
            
            documents = loader.load()
            
            if not documents:
                return {
                    'success': False,
                    'error': 'No transcript available for this video'
                }
            
            # Get video title
            video_title = documents[0].metadata.get('title', 'Unknown')
            logger.info(f"Video title: {video_title}")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Create vector store
            persist_directory = f"./youtube_db/{video_id}"
            os.makedirs(persist_directory, exist_ok=True)
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_directory,
                collection_name=f"youtube_{video_id}"
            )
            
            # Store for later use
            self.vectorstores[video_id] = vectorstore
            
            return {
                'success': True,
                'title': video_title,
                'transcript_length': len(documents[0].page_content),
                'chunks_created': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def ask_question(self, question: str, video_id: str) -> Dict:
        """Ask a question about the video."""
        try:
            if not self.llm:
                return {
                    'answer': 'Ollama is not running. Please start it with: ollama serve',
                    'sources': []
                }
            
            # Get vector store
            if video_id not in self.vectorstores:
                # Try to load from disk
                persist_directory = f"./youtube_db/{video_id}"
                if os.path.exists(persist_directory):
                    vectorstore = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=self.embeddings,
                        collection_name=f"youtube_{video_id}"
                    )
                    self.vectorstores[video_id] = vectorstore
                else:
                    return {
                        'answer': 'Video not loaded. Please load the video first.',
                        'sources': []
                    }
            
            vectorstore = self.vectorstores[video_id]
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Create prompt template
            template = """You are a helpful assistant answering questions about a YouTube video transcript.

Use the following pieces of context from the video transcript to answer the question.
If you don't know the answer based on the context, say that you don't know.
Keep your answer concise and based only on the provided context.

Context from video:
{context}

Question: {question}

Answer based on the video:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            # Get answer
            result = qa_chain.invoke({"query": question})
            
            # Extract sources
            sources = []
            for doc in result.get('source_documents', [])[:3]:
                sources.append({
                    'content': doc.page_content[:200] + "...",
                    'metadata': doc.metadata
                })
            
            return {
                'answer': result.get('result', 'No answer generated'),
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                'answer': f'Error: {str(e)}',
                'sources': []
            }

