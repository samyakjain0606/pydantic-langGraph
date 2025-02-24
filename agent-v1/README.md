# Pydantic LangGraph Investment Analysis Agent

An intelligent agent powered by Claude 3.5 Sonnet (via AWS Bedrock) that performs investment analysis by crawling websites, parsing PDFs, and generating detailed investment reports.

## Features

- **Web Crawling**: Extracts information from financial websites and company pages
- **PDF Parsing**: Processes financial documents and reports using LlamaParse
- **Report Generation**: Creates detailed investment analysis reports with multiple sections
- **Word Count Verification**: Ensures reports meet minimum length requirements
- **Logging**: Comprehensive logging system for debugging and monitoring

## Prerequisites

- Python 3.7+
- AWS Bedrock access configured
- Required API keys:
  - LLAMA_API_KEY (for PDF parsing)
  - SERP_API_KEY (for web search - currently disabled)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pydantic-langGraph.git

# Navigate to project directory
cd pydantic-langGraph

# Install dependencies
pip install anthropic-bedrock
pip install pydantic
pip install httpx
pip install serpapi
pip install devtools
pip install crawl4ai
pip install wget
pip install llama-cloud-services
pip install llama-index
```

## Environment Setup

1. Create a `.env` file in the root directory:

```env
LLAMA_API_KEY=your_llama_api_key
SERP_API_KEY=your_serp_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-west-2
```

2. Configure AWS credentials for Bedrock access:
   - Ensure you have the necessary AWS permissions for Claude 3.5 Sonnet
   - Configure AWS CLI or use environment variables

## Usage Example

```python
from bedrock_agent import agent
import asyncio
from httpx import AsyncClient
from dataclasses import dataclass
import os

async def analyze_company():
    async with AsyncClient() as client:
        deps = Deps(
            client=client,
            serp_api_key=os.getenv('SERP_API_KEY'),
            llama_api_key=os.getenv('LLAMA_API_KEY')
        )
        
        # Example: Analyze Chambal Fertilizers
        result = await agent.run(
            'Parse https://www.bseindia.com/xml-data/corpfiling/AttachHis/example.pdf and ' +
            'crawl https://www.screener.in/company/CHAMBLFERT/consolidated/ ' +
            'to create a detailed investment analysis report',
            deps=deps
        )
        print(result.data.summary)

if __name__ == "__main__":
    asyncio.run(analyze_company())
```

## Tool Documentation

### 1. crawl_website(url: str)
Scrapes financial websites and converts content to markdown format.
```python
result = await agent.crawl_website("https://www.screener.in/company/STOCK/")
```

### 2. parse_pdf_url(input_pdf_url: str)
Downloads and extracts content from PDF documents, with special handling for BSE India URLs.
```python
result = await agent.parse_pdf_url("https://www.bseindia.com/xml-data/corpfiling/example.pdf")
```

### 3. count_words_of_report(summary: str)
Validates report length and provides word count metrics.
```python
result = await agent.count_words_of_report(report_text)
```

## Output Format

The agent generates a structured report containing:

- Company Overview
- Financial Analysis
- Future Guidance
- Investment Thesis
- Risk Factors
- Recommendation

Each report is guaranteed to be at least 1200 words and includes comprehensive analysis.

## Logging

Logs are stored in `pdf_parser.log` and include:
- PDF processing events
- Download status
- Parsing results
- Error messages

To view logs:
```bash
tail -f pdf_parser.log
```

## Error Handling

The agent includes robust error handling for:
- PDF download failures
- BSE India specific URL processing
- Parsing errors
- Network timeouts
- API rate limits

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Anthropic's Claude](https://www.anthropic.com/claude) for AI capabilities
- [LlamaIndex](https://www.llamaindex.ai/) for PDF parsing
- [Crawl4AI](https://github.com/crawl4ai/crawl4ai) for web crawling

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 