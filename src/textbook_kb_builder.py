import os
import json
from typing import List, Dict
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

class TextbookKnowledgeBase:
    """Build and query RAG knowledge base"""
    
    def __init__(self, persist_dir=None):
        self.persist_dir = persist_dir or config.VECTOR_DB_DIR
        
        print("🗄️  Initializing RAG Knowledge Base...")
        
        # Disable telemetry to avoid warnings
        import chromadb.utils.embedding_functions as embedding_functions
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=config.RAG_CONFIG["collection_name"],
            metadata={"description": "Textbook knowledge base"}
        )
        
        # Load embedding model
        print(f"📥 Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        print("✅ RAG system initialized\n")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page numbers"""
        print(f"📄 Extracting text from: {pdf_path}")
        
        chunks = []
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Processing pages")):
                text = page.extract_text()
                
                if text.strip():
                    # Split page into smaller chunks
                    page_chunks = self._split_into_chunks(
                        text,
                        chunk_size=config.RAG_CONFIG["chunk_size"],
                        overlap=config.RAG_CONFIG["chunk_overlap"]
                    )
                    
                    for chunk_idx, chunk_text in enumerate(page_chunks):
                        chunks.append({
                            'text': chunk_text,
                            'page': page_num + 1,
                            'chunk_id': f"p{page_num+1}_c{chunk_idx}",
                            'source': os.path.basename(pdf_path)
                        })
        
        print(f"✅ Extracted {len(chunks)} chunks from {total_pages} pages\n")
        return chunks
    
    def _split_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_space = chunk.rfind(' ')
                break_point = max(last_period, last_newline, last_space)
                
                if break_point > chunk_size * 0.5:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap
        
        return chunks
    
    def build_knowledge_base(self, pdf_path: str):
        """Build knowledge base from PDF"""
        print("🏗️  Building Knowledge Base\n")
        
        # Extract text chunks
        chunks = self.extract_text_from_pdf(pdf_path)
        
        # Generate embeddings and store
        print("🔄 Generating embeddings and storing in database...")
        
        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size), desc="Storing chunks"):
            batch = chunks[i:i + batch_size]
            
            texts = [c['text'] for c in batch]
            metadatas = [{
                'page': c['page'],
                'chunk_id': c['chunk_id'],
                'source': c['source']
            } for c in batch]
            ids = [c['chunk_id'] for c in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            # Store in ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"\n✅ Knowledge base built: {len(chunks)} chunks stored")
        print(f"📁 Saved to: {self.persist_dir}\n")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant chunks for a query"""
        if top_k is None:
            top_k = config.RAG_CONFIG["top_k"]
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        retrieved = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                retrieved.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'id': results['ids'][0][i]
                })
        
        return retrieved
    
    def get_context_for_question(self, question: str, top_k: int = None) -> str:
        """Get formatted context string"""
        retrieved = self.retrieve(question, top_k)
        
        if not retrieved:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            page = doc['metadata']['page']
            text = doc['text'][:300]
            context_parts.append(f"[Source {i}, Page {page}]: {text}")
        
        return "\n\n".join(context_parts)
    
    def get_collection_info(self):
        """Get information about the collection"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }


def main():
    """Build knowledge base from textbook"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build RAG knowledge base")
    parser.add_argument('--pdf', type=str, required=True,
                       help='Path to textbook PDF')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for vector DB')
    
    args = parser.parse_args()
    
    # Initialize KB
    kb = TextbookKnowledgeBase(persist_dir=args.output)
    
    # Build from PDF
    kb.build_knowledge_base(args.pdf)
    
    # Test retrieval
    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60)
    
    test_query = "What are the main topics in this book?"
    print(f"\nTest Query: {test_query}\n")
    
    results = kb.retrieve(test_query, top_k=2)
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Page: {result['metadata']['page']}")
        print(f"  Text: {result['text'][:200]}...")
        print(f"  Distance: {result['distance']:.3f}\n")


if __name__ == "__main__":
    main()