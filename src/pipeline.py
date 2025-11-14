from .chunkers import Chunker
from .docparser import DocParser
from glob import glob
import pandas as pd
import traceback
from pathlib import Path
from .doc_qa import AgenticQA
from .indexing import indexing
from langchain_core.documents import Document
import os
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def list_supported_files(inputPath, supported_extensions=[".pdf", ".txt"]):
    file_list = glob(f"{inputPath}/**/*", recursive=True)
    return [f for f in file_list if Path(f).suffix.lower() in supported_extensions]

def pipeline(inputPath, parser_name, chunking_strategy, retrieval_strategy, input_type='directory', cli=True, domain=None):
    """
    Process input data (directory of files or CSV) for RAG pipeline with domain support.
    
    Args:
        inputPath (str): Path to directory or CSV file.
        parser_name (str): Name of the parser to use for directory files.
        chunking_strategy (str): Strategy for chunking documents.
        retrieval_strategy (str): Strategy for retrieval.
        input_type (str): 'directory' for files or 'csv' for CSV input. Defaults to 'directory'.
        cli (bool): Whether running from CLI. Defaults to True.
        domain (str): Domain identifier for separate collections. Auto-detected if None.
    
    Returns:
        AgenticQA: Initialized QA system with indexed data, or None if failed.
    """
    try:
        # AUTO-DETECT DOMAIN if not specified
        if not domain:
            input_lower = str(inputPath).lower()
            if "islamic" in input_lower:
                domain = "islamic_texts"
            elif "medquad" in input_lower or "medical" in input_lower:
                domain = "medical_csv" if input_type == 'csv' else "medical_docs"
            elif "docs" in input_lower:
                domain = "medical_docs"
            else:
                domain = None
        
        logger.info(f"🎯 Processing with domain: {domain}")

        # STEP 1: CHECK IF COLLECTION EXISTS FIRST (EFFICIENT!)
        doc_indexing = indexing()
        collection_name = f"{domain}_Agentic_retrieval" if domain else "Agentic_retrieval"
        
        try:
            existing_collection = doc_indexing.chroma_client.get_collection(name=collection_name)
            logger.info(f"📚 Found existing collection '{collection_name}' with {existing_collection.count()} documents")
            logger.info("⚡ Skipping document processing - using existing embeddings!")
            
            # Create retriever from existing collection
            from langchain_community.vectorstores import Chroma
            vector_store = Chroma(
                client=doc_indexing.chroma_client, # Connection to ChromaDB
                collection_name=collection_name, # Which collection to use
                embedding_function=doc_indexing.embedding_function # How to embed queries
            )
            
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            
            logger.info(f"✅ Pipeline loaded existing collection in seconds!")
            return retriever
            
        except Exception:
            # Collection doesn't exist - proceed with processing
            logger.info(f"🔄 Collection '{collection_name}' not found - will create new one")
            logger.info("⏰ Starting document processing...")
            
            
        chunks = []

        if input_type == 'directory':
            # Existing logic for processing directory files
            logger.info(f"Indexing files in {inputPath}: {os.listdir(inputPath)}")
            parser = DocParser(parser_name=parser_name)
            chunker = Chunker(chunking_strategy)
            
            files_list = list_supported_files(inputPath)
            if not files_list:
                raise ValueError(f"No supported files found in {inputPath}")
            
            # ADD PROGRESS TRACKING
            total_files = len(files_list)
            logger.info(f"📚 Processing {total_files} files...")
            
            for i, file_path in enumerate(files_list, 1):
                try:
                    logger.info(f"📄 [{i}/{total_files}] Processing: {Path(file_path).name}")
                    condition = Path(file_path).stem.lower().replace('_', ' ')
                    
                    # Parse document
                    text_docs = parser.parse(file_path)
                    if not text_docs:
                        logger.warning(f"{file_path} returned no text docs")
                        continue
                    if not all(isinstance(doc, Document) for doc in text_docs):
                        logger.warning(f"{file_path} returned non-Document objects: {[type(doc) for doc in text_docs]}")
                        text_docs = [Document(page_content=str(doc), metadata={"source": file_path}) for doc in text_docs]
                    
                    # Chunk document
                    file_chunks = chunker.build_chunks(text_docs, source=file_path)
                    if not file_chunks:
                        logger.warning(f"{file_path} returned no chunks")
                        continue
                    
                    # Validate and set metadata
                    for chunk in file_chunks:
                        if not hasattr(chunk, 'metadata'):
                            chunk.metadata = {}
                        chunk.metadata['condition'] = condition
                        chunk.metadata['source'] = file_path
                        # Add domain to metadata
                        chunk.metadata['domain'] = domain or 'general'
                    
                    logger.info(f"✅ Created {len(file_chunks)} chunks from {Path(file_path).name}")
                    chunks.extend(file_chunks)
                
                except Exception as e:
                    logger.error(f"❌ Failed to process {file_path}: {str(e)}")
                    continue

        elif input_type == 'csv':
            # CSV processing logic
            logger.info(f"Processing CSV file: {inputPath}")
            if not os.path.isfile(inputPath) or not inputPath.endswith('.csv'):
                raise ValueError(f"{inputPath} is not a valid CSV file")
            
            df = pd.read_csv(inputPath)
            required_columns = {'question', 'answer', 'source', 'focus_area'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            logger.info(f"📊 Processing {len(df)} CSV rows...")
            
            for index, row in df.iterrows():
                try:
                    page_content = row['answer']
                    metadata = {
                        'question': row['question'],
                        'source': row['source'],
                        'focus_area': row['focus_area'],
                        'condition': 'medical',
                        'domain': domain or 'medical'  # Add domain to metadata
                    }
                    chunk = Document(page_content=page_content, metadata=metadata)
                    chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Failed to process row {index}: {str(e)}")
                    continue
        
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")
        
        logger.info(f"📊 Total chunks processed: {len(chunks)}")
        
        # Initialize QA system with DOMAIN-AWARE indexing
        #agentic_qa = AgenticQA()
        
        if chunks:
            # Use your enhanced indexing class with domain support
            doc_indexing = indexing()
            
            # Pass domain to create separate collections
            retriever = doc_indexing.index_documents(
                documents=chunks,
                domain=domain,  # THIS IS THE KEY CHANGE!
                top_k=5
            )
            
            #agentic_qa.run(retriever,domain=domain)
            logger.info(f"✅ Pipeline completed successfully for domain: {domain}")
        else:
            logger.warning("No valid documents were processed")
            
        
        return None
    
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {str(e)}")
        traceback.print_exc()
        return None
    