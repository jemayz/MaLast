import pymupdf4llm
import fitz
import os
import io
from pathlib import Path
from typing import List
from llama_parse import LlamaParse
from langchain_core.documents import Document
from dotenv import find_dotenv, load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

class DocParser:
    def __init__(self, parser_name):
        self.parser_name = parser_name
        self.assets_dir = "./parsed_assets/"
        self.parser_function_map = {
            "LlamaParse": self.with_LlamaParse,
            "pymupdf4llm": self.with_pymupdf4llm
        }
        self.parsing_function = self.parser_function_map.get(parser_name)
        if not self.parsing_function:
            raise ValueError(f"Unsupported parser: {parser_name}")
        
        # Ensure the save directory exists
        os.makedirs(self.assets_dir, exist_ok=True)

    def parse(self, file_path):
        text_docs = self.parsing_function(file_path)
        if self.parser_name == "LlamaParse":
            self.extract_images(file_path)
        return text_docs
    
    def with_LlamaParse(self, file_path):
        logger.info("Using LlamaParse to parse document...")
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=False
        )
        data = parser.load_data(file_path=file_path)
        text_docs = [Document(page_content=x.text, metadata={"source": file_path}) for x in data]
        logger.debug(f"LlamaParse extracted {len(text_docs)} documents from {file_path}")
        return text_docs
    
    def with_pymupdf4llm(self, file_path):
        logger.info(f"Using pymupdf4llm to parse {file_path}")
        try:
            output = pymupdf4llm.to_markdown(
                file_path, 
                write_images=True,
                image_path=self.assets_dir,
                extract_words=True,
                show_progress=False
            )
            # Convert output to Document objects
            if not output:
                logger.warning(f"No content extracted from {file_path}")
                return []
            
            text_docs = [
                Document(
                    page_content=x["text"].replace("-----", ""),
                    metadata={"source": file_path, "page": i}
                )
                for i, x in enumerate(output) if x.get("text")
            ]
            logger.debug(f"pymupdf4llm extracted {len(text_docs)} documents from {file_path}")
            return text_docs
        except Exception as e:
            logger.error(f"Error parsing {file_path} with pymupdf4llm: {str(e)}")
            return []

    # def extract_images(self, filepath):
    #     logger.info(f"Extracting images from {filepath}")
    #     try:
    #         doc = fitz.open(filepath)
    #         save_dir = Path(self.assets_dir)

    #         for p in range(len(doc)):
    #             page = doc[p]
    #             for i, img in enumerate(page.get_images(), start=1):
    #                 xref = img[0]
    #                 base_image = doc.extract_image(xref)
    #                 image_bytes = base_image["image"]
    #                 pil_image = Image.open(io.BytesIO(image_bytes))
    #                 image_name = f"{save_dir.joinpath(Path(filepath).stem)}_{p}_image{i}.png"
    #                 pil_image.save(image_name)
    #                 logger.debug(f"Saved image: {image_name}")
    #     except Exception as e:
    #         logger.error(f"Error extracting images from {filepath}: {str(e)}")