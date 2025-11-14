import glob
import time
import base64
import logging
import uuid6
from pathlib import Path
from typing import List
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.image_dir = "./parsed_assets/"
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0
        )

    @staticmethod
    def retry_with_delay(func, *args, delay=2, retries=30, **kwargs):
        """
        Helper method to retry a function call with a delay.
        """
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(delay)
        logger.error("Exceeded maximum retries.")
        return None

    def encode_image(self, image_path):
        """Encode image to a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None

    def image_summarize(self, img_base64):
        """Generate image summary using LLM."""
        if not img_base64:
            return "Error: Invalid image base64"
       
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
                    These summaries will be embedded and used to retrieve the raw image. \
                    Give a concise summary of the image that is well optimized for retrieval."""
       
        try:
            msg = self.llm.invoke([
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    ]
                )
            ])
            return msg.content if msg else "Error: No response from LLM"
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Error: Failed to generate summary"

    def get_image_summaries(self):
        """
        Generate summaries for images in the directory.
        """
        image_summaries = []
        image_paths = sorted(glob.glob(f"{self.image_dir}*.png"))
       
        for img_path in image_paths:
            logger.info(f"Processing image: {img_path}")
            base64_image = self.encode_image(img_path)
            if not base64_image:
                image_summaries.append("Error: Could not encode image")
                continue

            summary = self.retry_with_delay(self.image_summarize, base64_image)
            if not summary:
                summary = "Error: Failed to generate summary"
           
            image_summaries.append(summary)
            time.sleep(1)  # Delay to prevent API throttling

        return image_summaries

    def get_image_documents(self) -> List[Document]:
        """
        Extract images and generate corresponding Document objects.
        """
        image_documents = []
        image_summaries = self.get_image_summaries()
        image_paths = sorted(glob.glob(f"{self.image_dir}*.png"))
       
        for summary, image_path in zip(image_summaries, image_paths):
            doc = Document(
                page_content=summary,
                metadata={"source": Path(image_path).name},
                id=str(uuid6.uuid6()),
            )
            image_documents.append(doc)
            logger.info(f"Generated document for {image_path}: {summary[:100]}...")

        return image_documents