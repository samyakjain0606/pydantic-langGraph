from anthropic import AsyncAnthropicBedrock
from pydantic_ai.models.anthropic import AnthropicModel
# from pydantic_ai import Agent
from pydantic_ai import Agent, ModelRetry, RunContext
from dataclasses import dataclass
import asyncio
import os
from pydantic import BaseModel
from httpx import AsyncClient
import serpapi
from devtools import debug
from crawl4ai import *
import wget
import requests
from typing import Optional
import time
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import tempfile
import aiohttp
import logging
import sys

anthropic_bedrock_client = AsyncAnthropicBedrock(aws_region='us-west-2')

model = AnthropicModel(
    model_name='anthropic.claude-3-5-sonnet-20241022-v2:0',
    anthropic_client=anthropic_bedrock_client
)

@dataclass
class Deps:
    client: AsyncClient
    serp_api_key: str | None
    llama_api_key: str | None

class MyModel(BaseModel):
    city: str
    country: str
    calculated_result: int

class SummaryResponse(BaseModel):
    summary: str

print(f'Using model: {model}')
# agent = Agent(model, result_type=MyModel)
agent = Agent(model,
    system_prompt=(
        'You are websearch agent for investment analysis. Your primary tasks are:'
        '\n1. Search and gather information using crawl_website and parse_pdf_url and count_words_of_report'
        '\n2. Create REPORTS as requested and gather insightful information'
    ),
    deps_type=Deps,
    result_type=SummaryResponse,
    retries=10,
)

# @agent.tool 
# async def web_search(ctx: RunContext[Deps], query: str) -> str:

#     """
#     Get information from the web.
#     Args:
#         ctx: The context
#         query: Google search query to make
#     """
#     if ctx.deps.serp_api_key is None: 
#         return "Please provide serp api key, otherwise web search is not possible"
#     params = {
#         "engine": "google",
#         "q": query
#         # "api_key": ctx.deps.serp_api_key
#     }
#     client = serpapi.Client(api_key=ctx.deps.serp_api_key)
#     results = client.search(params)
#     organic_results = results["organic_results"]
#     return organic_results

@agent.tool
async def count_words_of_report(ctx: RunContext[Deps], summary: str) -> dict[str, any]:
    """
    Count the number of words in the summary and verify if it matches the required count.
    Args:
        ctx: The context
        summary: The text to analyze
    Returns:
        Dictionary containing word count, match status, and the summary
    """
    words = summary.split()
    word_count = len(words)
    print(f'Current word count: {word_count}')
    
    return {
        "word_count": word_count,
        "summary": summary
    }

@agent.tool
async def crawl_website(ctx: RunContext[Deps], url: str) -> str:
    """
    Website to crawl to get information
    Args:
        ctx: The context
        url: The url to crawl
    Returns:
        Website scraped in form of string
    """
    crawler_cfg = CrawlerRunConfig(
        exclude_external_images=True,
        exclude_external_links=True,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,config=crawler_cfg
        )
        return (result.markdown)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_parser.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('pdf_parser')

@agent.tool
async def parse_pdf_url(ctx: RunContext[Deps], input_pdf_url: str) -> str:
    """
    Parse a PDF file from a given URL using LlamaParse
    Args:
        ctx: The context
        input_pdf_url: URL of the PDF to parse
    Returns:
        Parsed PDF content or error message
    """
    if ctx.deps.llama_api_key is None:
        logger.error("LlamaIndex API key not provided")
        return "Please provide LlamaIndex API key for PDF parsing"
    
    try:
        logger.info(f"Starting to process PDF from URL: {input_pdf_url}")
        
        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            # Download the PDF with browser-like headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            logger.info("Attempting to download PDF...")
            async with aiohttp.ClientSession() as session:
                async with session.get(input_pdf_url, headers=headers, allow_redirects=True) as response:
                    if response.status != 200:
                        # If it's a BSE India URL, try to extract the actual PDF URL
                        if 'bseindia.com' in input_pdf_url:
                            logger.info("BSE India URL detected, attempting direct download...")
                            pdf_name = input_pdf_url.split('Pname=')[-1]
                            direct_url = f'{pdf_name}'
                            logger.info(f"Trying direct BSE URL: {direct_url}")
                            async with session.get(direct_url, headers=headers) as bse_response:
                                if bse_response.status != 200:
                                    logger.error(f"Failed to download PDF from BSE. Status: {bse_response.status}")
                                    return f"Failed to download PDF from BSE: {bse_response.status}"
                                content = await bse_response.read()
                                logger.info("Successfully downloaded PDF from BSE")
                        else:
                            logger.error(f"Failed to download PDF. Status: {response.status}")
                            return f"Failed to download PDF: {response.status}"
                    else:
                        content = await response.read()
                        logger.info("Successfully downloaded PDF")
                    
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                    logger.info(f"PDF saved to temporary file: {temp_file_path}")

        # Set up parser
        logger.info("Initializing LlamaParse...")
        parser = LlamaParse(
            api_key=ctx.deps.llama_api_key,
            result_type="markdown"
        )

        # Parse the PDF
        try:
            logger.info("Starting PDF parsing...")
            parsed_content = await parser.aload_data(temp_file_path)
            if parsed_content and len(parsed_content) > 0:
                logger.info(f"Successfully parsed PDF. Content length: {len(str(parsed_content))} characters")
                return parsed_content
            else:
                logger.warning("No content parsed from PDF")
                return "No content parsed from PDF"
        except Exception as parse_error:
            logger.error(f"Error during parsing: {str(parse_error)}", exc_info=True)
            return f"Error during parsing: {str(parse_error)}"

    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}", exc_info=True)
        return f"Error parsing PDF: {str(e)}"
    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")

# if __name__ == '__main__':
#     result = agent.run_sync('Calculate 2+3+5+8')
#     print(result.data)
#     print(result.usage())  

async def main():
    async with AsyncClient() as client:
        serp_api_key = os.getenv('SERP_API_KEY')
        llama_api_key = os.getenv('LLAMA_API_KEY')
        deps = Deps(
            client=client, 
            serp_api_key=serp_api_key,
            llama_api_key=llama_api_key
        )
        result = await agent.run(
            'Parse this pdf url https://www.bseindia.com/xml-data/corpfiling/AttachHis/3e629215-d629-4d24-9626-0d9a1561cc80.pdf and also crawl https://www.screener.in/company/CHAMBLFERT/consolidated/ to get more information PLEASE DO STUFF ONE BY ONE!'
            'Then create a detailed report of more than 1200 words of the key findings, have multiple sections, like future guidance, historical financial data, what can be the thesis, should the stock be looked upon from a investment perspective. MAKE SURE THE REPORT IS OF MORE THAN 1200 WORDS', 
            deps=deps
        )
        debug(result)
        print('\nREPORT:', result.data.summary)


if __name__ == '__main__':
    asyncio.run(main())