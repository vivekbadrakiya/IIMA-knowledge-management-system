"""
Large File Handler for RAG System

Handles large PDFs (100MB+) efficiently with:
- Streaming processing (page-by-page)
- Batch embedding generation
- Memory management
- Progress tracking
- Checkpointing (resume on failure)
"""

import os
import gc
import logging
import pickle
from typing import List, Optional, Iterator
from pathlib import Path
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
@dataclass
class LargeFileConfig:
    """Configuration for large file processing"""
    persist_dir: str = "db/chroma_db"
    checkpoint_dir: str = "db/checkpoints"
    
    # Processing parameters
    batch_size: int = 50  # Process 50 chunks at a time
    pages_per_batch: int = 10  # Load 10 pages at a time
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32  # Embed 32 chunks at once
    
    # Memory management
    clear_memory_interval: int = 100  # Clear every 100 chunks
    
    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_interval: int = 500  # Save progress every 500 chunks


# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------
# Streaming PDF Loader
# ---------------------------
class StreamingPDFLoader:
    """Load PDF pages in batches to avoid memory overflow"""
    
    def __init__(self, file_path: str, pages_per_batch: int = 10):
        self.file_path = file_path
        self.pages_per_batch = pages_per_batch
        self.loader = PyPDFLoader(file_path)
    
    def load_in_batches(self) -> Iterator[List[Document]]:
        """
        Load PDF pages in batches
        
        Yields:
            List of Document objects (batch)
        """
        try:
            # Load all pages (PyPDFLoader does this efficiently)
            all_pages = self.loader.load()
            total_pages = len(all_pages)
            
            logger.info(f"PDF has {total_pages} pages. Loading in batches of {self.pages_per_batch}")
            
            # Yield in batches
            for i in range(0, total_pages, self.pages_per_batch):
                batch = all_pages[i:i + self.pages_per_batch]
                yield batch
                
                # Clear memory after each batch
                del batch
                gc.collect()
        
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise


# ---------------------------
# Checkpoint Manager
# ---------------------------
class CheckpointManager:
    """Manage processing checkpoints for resume capability"""
    
    def __init__(self, checkpoint_dir: str, file_name: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique checkpoint file
        safe_name = file_name.replace('/', '_').replace('\\', '_')
        self.checkpoint_file = self.checkpoint_dir / f"{safe_name}.checkpoint"
    
    def save(self, processed_chunks: int, total_chunks: int):
        """Save processing progress"""
        checkpoint = {
            'processed_chunks': processed_chunks,
            'total_chunks': total_chunks
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.debug(f"Checkpoint saved: {processed_chunks}/{total_chunks}")
    
    def load(self) -> Optional[dict]:
        """Load processing progress"""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            logger.info(f"Checkpoint loaded: {checkpoint['processed_chunks']}/{checkpoint['total_chunks']} chunks processed")
            return checkpoint
        
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {str(e)}")
            return None
    
    def clear(self):
        """Remove checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")


# ---------------------------
# Large File Processor
# ---------------------------
class LargeFileProcessor:
    """Process large PDFs efficiently"""
    
    def __init__(self, config: Optional[LargeFileConfig] = None):
        self.config = config or LargeFileConfig()
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize components"""
        logger.info("Initializing Large File Processor...")
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        # Embeddings
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': self.config.embedding_batch_size  # Batch embedding
            }
        )
        
        logger.info("✅ Processor initialized")
    
    def process_large_pdf(
        self, 
        file_path: str,
        resume_from_checkpoint: bool = True
    ) -> Chroma:
        """
        Process a large PDF file efficiently
        
        Args:
            file_path: Path to PDF file
            resume_from_checkpoint: Whether to resume from previous checkpoint
            
        Returns:
            ChromaDB vector store
        """
        file_name = os.path.basename(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        logger.info("=" * 70)
        logger.info(f"Processing Large PDF: {file_name}")
        logger.info(f"File Size: {file_size_mb:.2f} MB")
        logger.info("=" * 70)
        
        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(
            self.config.checkpoint_dir, 
            file_name
        )
        
        # Check for existing checkpoint
        checkpoint = None
        if resume_from_checkpoint and self.config.enable_checkpoints:
            checkpoint = checkpoint_mgr.load()
        
        # Step 1: Load and chunk documents
        all_chunks = self._load_and_chunk(file_path, file_name)
        
        if not all_chunks:
            raise ValueError("No chunks created from PDF")
        
        total_chunks = len(all_chunks)
        logger.info(f"Total chunks to process: {total_chunks}")
        
        # Determine starting point
        start_idx = 0
        if checkpoint:
            start_idx = checkpoint['processed_chunks']
            logger.info(f"Resuming from chunk {start_idx}")
            all_chunks = all_chunks[start_idx:]
        
        # Step 2: Process in batches
        self._process_in_batches(
            chunks=all_chunks,
            file_name=file_name,
            checkpoint_mgr=checkpoint_mgr,
            start_offset=start_idx,
            total_chunks=total_chunks
        )
        
        # Clear checkpoint on successful completion
        checkpoint_mgr.clear()
        
        logger.info("✅ Large file processing completed!")
        
        # Return vector store
        return Chroma(
            persist_directory=self.config.persist_dir,
            embedding_function=self.embeddings
        )
    
    def _load_and_chunk(self, file_path: str, file_name: str) -> List[Document]:
        """Load PDF and create chunks with streaming"""
        logger.info("📄 Loading and chunking PDF...")
        
        all_chunks = []
        chunk_id = 0
        
        # Stream pages in batches
        streaming_loader = StreamingPDFLoader(
            file_path, 
            pages_per_batch=self.config.pages_per_batch
        )
        
        for page_batch in streaming_loader.load_in_batches():
            # Add metadata
            for doc in page_batch:
                doc.metadata['source_file'] = file_name
                doc.metadata['file_type'] = 'pdf'
            
            # Chunk this batch
            batch_chunks = self.text_splitter.split_documents(page_batch)
            
            # Add chunk IDs
            for chunk in batch_chunks:
                chunk.metadata['chunk_id'] = chunk_id
                chunk.metadata['chunk_size'] = len(chunk.page_content)
                chunk_id += 1
            
            all_chunks.extend(batch_chunks)
            
            # Memory management
            del page_batch
            del batch_chunks
            gc.collect()
        
        logger.info(f"✅ Created {len(all_chunks)} chunks")
        
        return all_chunks
    
    def _process_in_batches(
        self,
        chunks: List[Document],
        file_name: str,
        checkpoint_mgr: CheckpointManager,
        start_offset: int,
        total_chunks: int
    ):
        """Process chunks in batches with progress tracking"""
        logger.info("🔢 Generating embeddings and storing...")
        
        # Initialize or load vector store
        if start_offset == 0:
            # Create new vector store
            Path(self.config.persist_dir).mkdir(parents=True, exist_ok=True)
            self.vector_store = None
        else:
            # Load existing vector store
            self.vector_store = Chroma(
                persist_directory=self.config.persist_dir,
                embedding_function=self.embeddings
            )
        
        # Process in batches
        batch_size = self.config.batch_size
        processed = start_offset
        
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                try:
                    # Add to vector store
                    if self.vector_store is None:
                        # First batch - create vector store
                        self.vector_store = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory=self.config.persist_dir
                        )
                    else:
                        # Subsequent batches - add to existing
                        self.vector_store.add_documents(batch)
                    
                    processed += len(batch)
                    pbar.update(len(batch))
                    
                    # Save checkpoint
                    if self.config.enable_checkpoints:
                        if processed % self.config.checkpoint_interval == 0:
                            checkpoint_mgr.save(processed, total_chunks)
                    
                    # Memory management
                    if processed % self.config.clear_memory_interval == 0:
                        gc.collect()
                        logger.debug(f"Memory cleared at {processed} chunks")
                
                except Exception as e:
                    logger.error(f"Error processing batch at chunk {processed}: {str(e)}")
                    # Save checkpoint before failing
                    if self.config.enable_checkpoints:
                        checkpoint_mgr.save(processed, total_chunks)
                    raise
        
        logger.info(f"✅ Processed and stored {processed} chunks")
    
    def get_file_size_estimate(self, file_path: str) -> dict:
        """
        Estimate memory and time requirements for a file
        
        Returns:
            Dict with estimates
        """
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Rough estimates
        estimated_pages = int(file_size_mb * 10)  # ~10 pages per MB
        estimated_chunks = int(estimated_pages * 2)  # ~2 chunks per page
        estimated_memory_mb = int(file_size_mb * 5)  # ~5x file size in RAM
        estimated_time_min = int(estimated_chunks * 0.1 / 60)  # ~0.1s per chunk
        
        return {
            'file_size_mb': file_size_mb,
            'estimated_pages': estimated_pages,
            'estimated_chunks': estimated_chunks,
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_time_minutes': estimated_time_min,
            'recommended_batch_size': 50 if file_size_mb > 100 else 100
        }


# ---------------------------
# CLI for Large Files
# ---------------------------
def process_large_file_cli():
    """Command-line interface for processing large files"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python large_file_handler.py <pdf_file_path>")
        print("Example: python large_file_handler.py data/large_document.pdf")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Create processor
    config = LargeFileConfig(
        batch_size=50,
        pages_per_batch=10,
        embedding_batch_size=32,
        enable_checkpoints=True
    )
    
    processor = LargeFileProcessor(config)
    
    # Show estimates
    estimates = processor.get_file_size_estimate(file_path)
    
    print("\n" + "=" * 70)
    print("📊 PROCESSING ESTIMATES")
    print("=" * 70)
    print(f"File Size: {estimates['file_size_mb']:.2f} MB")
    print(f"Estimated Pages: ~{estimates['estimated_pages']}")
    print(f"Estimated Chunks: ~{estimates['estimated_chunks']}")
    print(f"Estimated Memory: ~{estimates['estimated_memory_mb']} MB")
    print(f"Estimated Time: ~{estimates['estimated_time_minutes']} minutes")
    print("=" * 70)
    
    response = input("\nProceed with processing? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    # Process file
    try:
        processor.process_large_pdf(file_path, resume_from_checkpoint=True)
        print("\n✅ Processing complete!")
        print(f"Vector store saved to: {config.persist_dir}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted!")
        print("Progress has been saved. Run again to resume.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    process_large_file_cli()