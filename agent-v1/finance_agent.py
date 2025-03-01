from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import asyncio
import serpapi
import operator
import aiohttp
import tempfile
import os
import logging
from llama_cloud_services import LlamaParse
from IPython.display import Image, display

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('research_agent')

### Tools
def crawl_webpage(url: str) -> str:
    """Crawls a webpage and returns its content.
    
    Args:
        url: The URL to crawl
    """
    crawler_cfg = CrawlerRunConfig(
        exclude_external_images=True,
        exclude_external_links=True,
    )
    
    # Create an event loop and run the async function
    loop = asyncio.new_event_loop()
    try:
        async def _crawl():
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=crawler_cfg)
                return result.markdown
        
        return loop.run_until_complete(_crawl())
    finally:
        loop.close()

async def parse_pdf(pdf_url: str) -> str:
    """Parses a PDF from a URL and returns its content.
    
    Args:
        pdf_url (str): URL of the PDF to parse
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    if response.status != 200:
                        return f"Failed to download PDF: {response.status}"
                    content = await response.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name

            parser = LlamaParse(
                api_key="llx-9XpgmTKvNL1RyaU0nFIYD7vmYAT8BhQdQ0SsIhubyZwrFw26",
                result_type="markdown"
            )
            parsed_content = await parser.aload_data(temp_file_path)
            return parsed_content if parsed_content else "No content parsed from PDF"

    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        return f"Error parsing PDF: {str(e)}"
    finally:
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")

def sync_parse_pdf(pdf_url: str) -> str:
    """Synchronous wrapper for parse_pdf function"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(parse_pdf(pdf_url))
    finally:
        loop.close()

def web_search(query: str) -> str:
    """Performs a web search using SerpAPI.
    
    Args:
        query: Search query string
    """
    params = {
        "engine": "google",
        "q": query
    }
    client = serpapi.Client(api_key="d3a459db117ccb0e8f08033fcd4a6c55bc075ec90bf594cede01db523cd7731d")
    results = client.search(params)
    return str(results.get("organic_results", []))

def count_words(text: str) -> int:
    """Counts words in a text.
    
    Args:
        text (str): Text to analyze
    """
    return len(text.split())

### State
def merge_dicts(dict1, dict2):
    """Custom merge function for dictionaries"""
    if not dict1:
        return dict2
    if not dict2:
        return dict1
    return {**dict1, **dict2}

class ResearchState(MessagesState):
    context: Annotated[dict, merge_dicts] = {}  # Store extracted data
    company_info: Optional[str] = None
    business_model: Optional[str] = None
    revenue_sources: Optional[str] = None
    financial_analysis: Optional[str] = None
    growth_triggers: Optional[list] = None
    capex_analysis: Optional[str] = None
    market_position: Optional[str] = None
    risk_analysis: Optional[list] = None
    investment_recommendation: Optional[list] = None
    final_report: Optional[str] = None

### LLM Setup
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    model_kwargs=dict(temperature=0),
)

# Bind tools to LLM
tools = [crawl_webpage, sync_parse_pdf]
llm_with_tools = llm.bind_tools(tools)

### Nodes
def data_collector(state: ResearchState):
    """Initial node that collects all data from tools once"""
    logger.info("🔄 Starting data collection node")
    logger.info(state["messages"])
    sys_msg = SystemMessage(content="""You are a data collection assistant. 
    Use the tools to gather information about Time Technoplast:
    1. Use parse_pdf to extract information from the earnings transcript
    2. Use crawl_webpage to get financial data from screener.in
    
    IMPORTANT: After you have used both tools and gathered all the information, 
    provide a comprehensive summary of all collected data. Do not use any more tools after summarizing.""")
    
    # Return just the LLM response - tools will be handled by the graph structure
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    
    # Store the collected data in context if it's a summary (not a tool call)
    if not response.additional_kwargs.get('tool_calls'):
        logger.info("✅ Data collection complete, moving to analysis")
        
        # Collect all tool responses
        tool_responses = {}
        count_parse_pdf = 0
        count_crawl = 0
        for msg in state["messages"]:
            if hasattr(msg, 'name') and hasattr(msg, 'content'):  # Check if it's a tool message
                tool_responses[msg.name] = msg.content
                print(msg.name)
                if msg.name == 'crawl_webpage':
                    count_crawl = 1
                if msg.name == 'sync_parse_pdf':
                    count_parse_pdf = 1
                if count_crawl>0 and count_parse_pdf>0:
                    print('here count crawl parse pdf')
                    return {
                        "messages": [response],
                        "context": {
                            "tool_responses": tool_responses,  # Raw tool data
                            "collected_data": response.content  # LLM's summary
                        }
                    }
                    
        
        return {
            "messages": [response],
            "context": {
                "tool_responses": tool_responses,  # Raw tool data
                "collected_data": response.content  # LLM's summary
            }
        }
    
    logger.info("🔧 Using tools to collect data")
    return {"messages": [response]}

def company_info_analyst(state: ResearchState):
    """Analyzes basic company information"""
    logger.info("🏢 Starting company info analysis")
    sys_msg = SystemMessage(content="""You are a specialized company information analyst. Analyze the provided data and create a structured overview of the company with these specific sections:

1. Company Background
   - Year founded
   - Key milestones
   - Corporate structure

2. Core Business Areas
   - Main product lines
   - Key technologies/patents
   - Manufacturing facilities

3. Geographic Presence
   - Key markets
   - Distribution network
   - Export presence

4. Management Overview
   - Key management personnel
   - Notable expertise

Be precise and data-focused. Avoid generic statements. Use specific numbers and facts from the provided data.
Limit your response to 500 words and maintain a professional analytical tone.""")
    
    # Pass both raw tool responses and the summary
    context_data = {
        "tool_responses": state["context"]["tool_responses"],
        "summary": state["context"]["collected_data"]
    }
    
    messages = [sys_msg, HumanMessage(content=str(context_data))]
    response = llm.invoke(messages)
    logger.info("✅ Company info analysis complete")
    return {"company_info": response.content}

def business_model_analyst(state: ResearchState):
    """Analyzes business model"""
    logger.info("💼 Starting business model analysis")
    sys_msg = SystemMessage(content="""You are a business model analysis specialist. Create a detailed analysis of the company's business model with these specific sections:

1. Value Proposition
   - Core offerings
   - Customer segments
   - Unique selling points

2. Operational Structure
   - Manufacturing process
   - Supply chain overview
   - Distribution channels

3. Competitive Advantages
   - Technology differentiators
   - Cost advantages
   - Market positioning

4. Strategic Partnerships
   - Key collaborations
   - Integration with suppliers/customers

Focus on quantifiable metrics where possible. Highlight specific examples that demonstrate the business model's effectiveness.
Limit response to 500 words and maintain an analytical perspective.""")
    
    messages = [sys_msg, HumanMessage(content=str(state["context"]))]
    response = llm.invoke(messages)
    logger.info("✅ Business model analysis complete")
    return {"business_model": response.content}

def revenue_analyst(state: ResearchState):
    """Analyzes revenue sources"""
    logger.info("💰 Starting revenue analysis")
    sys_msg = SystemMessage(content="""You are a revenue analysis expert. Provide a comprehensive breakdown of the company's revenue structure:

1. Revenue Segmentation
   - Product-wise breakdown
   - Geographic distribution
   - Customer segment contribution

2. Revenue Trends
   - YoY growth rates
   - Segment-wise growth
   - Seasonal patterns

3. Revenue Quality Analysis
   - Revenue concentration
   - Recurring vs one-time
   - Contract nature (long-term/short-term)

4. Pricing Power
   - Pricing trends
   - Margin analysis by segment

Use specific numbers and percentages. Compare with historical data where available.
Create clear insights about revenue sustainability and growth.""")
    
    messages = [sys_msg, HumanMessage(content=str(state["context"]))]
    response = llm.invoke(messages)
    logger.info("✅ Revenue analysis complete")
    return {"revenue_sources": response.content}

def financial_analyst(state: ResearchState):
    """Analyzes financial metrics"""
    logger.info("📊 Starting financial metrics analysis")
    sys_msg = SystemMessage(content="""You are a financial analysis expert. Conduct a thorough financial analysis with these specific sections:

1. Profitability Metrics
   - Gross margins and trends
   - EBITDA margins
   - Net profit margins
   - Return ratios (ROE, ROCE)

2. Balance Sheet Strength
   - Working capital analysis
   - Debt metrics and coverage
   - Asset utilization ratios
   - Capital structure

3. Cash Flow Analysis
   - Operating cash flow trends
   - Free cash flow generation
   - Cash conversion cycle
   - Working capital management

4. Key Financial Indicators
   - Liquidity ratios
   - Solvency metrics
   - Efficiency ratios

Present specific numbers, ratios, and their trends. Compare with industry standards where relevant.
Highlight both strengths and areas of concern.""")
    
    messages = [sys_msg, HumanMessage(content=str(state["context"]))]
    response = llm.invoke(messages)
    logger.info("✅ Financial analysis complete")
    return {"financial_analysis": response.content}

def growth_analyst(state: ResearchState):
    """Analyzes growth triggers"""
    logger.info("📈 Starting growth triggers analysis")
    sys_msg = SystemMessage(content="""You are a growth analysis specialist. Identify and analyze key growth drivers:

1. Organic Growth Drivers
   - Market expansion opportunities
   - Product development pipeline
   - Capacity expansion plans
   - Technology upgrades

2. External Growth Factors
   - Industry tailwinds
   - Policy support
   - Export opportunities
   - Market consolidation potential

3. Operational Growth Levers
   - Operating leverage
   - Efficiency improvements
   - Cost optimization initiatives

4. Future Growth Catalysts
   - New market entries
   - Product launches
   - Strategic initiatives

Quantify growth potential where possible. Provide specific timelines and metrics for growth initiatives.""")
    
    messages = [sys_msg, HumanMessage(content=str(state["context"]))]
    response = llm.invoke(messages)
    logger.info("✅ Growth analysis complete")
    return {"growth_triggers": response.content}

def capex_analyst(state: ResearchState):
    """Analyzes CAPEX and order books"""
    logger.info("🏗️ Starting CAPEX analysis")
    sys_msg = SystemMessage(content="""You are a CAPEX and order book analysis specialist. Provide detailed analysis of:

1. CAPEX Plans
   - Ongoing projects
   - Planned investments
   - Expansion timelines
   - Technology upgrades

2. Order Book Analysis
   - Current order book value
   - Order book composition
   - Execution timeline
   - Client concentration

3. Funding Structure
   - Source of funds
   - Cost of capital
   - Debt-equity mix
   - Return projections

4. Impact Analysis
   - Capacity addition
   - Revenue potential
   - Margin implications
   - Payback periods

Use specific numbers and timelines. Analyze the quality of CAPEX and its strategic alignment.""")
    
    messages = [sys_msg, HumanMessage(content=str(state["context"]))]
    response = llm.invoke(messages)
    logger.info("✅ CAPEX analysis complete")
    return {"capex_analysis": response.content}

def market_analyst(state: ResearchState):
    """Analyzes market position and tailwinds"""
    logger.info("🌐 Starting market position analysis")
    sys_msg = SystemMessage(content="""You are a market analysis expert. Evaluate the company's market position:

1. Market Share Analysis
   - Overall market position
   - Segment-wise share
   - Competitive ranking
   - Market share trends

2. Industry Analysis
   - Market size and growth
   - Demand drivers
   - Supply dynamics
   - Regulatory environment

3. Competitive Landscape
   - Key competitors
   - Competitive advantages
   - Entry barriers
   - Threat assessment

4. Market Opportunities
   - Untapped segments
   - Geographic expansion
   - Product gaps
   - Market consolidation

Provide specific market sizes, growth rates, and competitive positions.
Analyze both short-term and long-term market dynamics.""")
    
    messages = [sys_msg, HumanMessage(content=str(state["context"]))]
    response = llm.invoke(messages)
    logger.info("✅ Market analysis complete")
    return {"market_position": response.content}

def risk_analyst(state: ResearchState):
    """Analyzes risks"""
    logger.info("⚠️ Starting risk analysis")
    sys_msg = SystemMessage(content="""You are a risk assessment specialist. Identify and analyze key risks:

1. Operational Risks
2. Financial Risks
3. Market Risks

Rate each risk category (High/Medium/Low). Provide specific examples and mitigation strategies.""")
    
    messages = [sys_msg, HumanMessage(content=str(state["context"]))]
    response = llm.invoke(messages)
    logger.info("✅ Risk analysis complete")
    return {"risk_analysis": response.content}

def investment_analyst(state: ResearchState):
    """Provides investment recommendation"""
    logger.info("💡 Starting investment recommendation analysis")
    sys_msg = SystemMessage(content="""You are an investment recommendation specialist. Provide a comprehensive investment analysis:

1. Investment Thesis
   - Key investment merits
   - Growth catalysts
   - Competitive advantages
   - Management quality

2. Valuation Analysis
   - Current valuations
   - Peer comparison
   - Historical trends
   - Fair value assessment

3. Return Potential
   - Expected growth rates
   - Margin expansion
   - Multiple re-rating
   - Dividend potential

4. Investment Risks
   - Key concerns
   - Risk-reward ratio
   - Investment horizon
   - Entry/exit points

Conclude with a clear recommendation (Buy/Hold/Sell) and target price range.
Provide specific triggers for reviewing the recommendation.""")
    
    messages = [sys_msg, HumanMessage(content=str(state["context"]))]
    response = llm.invoke(messages)
    logger.info("✅ Investment recommendation complete")
    return {"investment_recommendation": response.content}

def report_compiler(state: ResearchState):
    """Compiles final report from all analyses"""
    logger.info("📝 Starting final report compilation")
    sections = [
        ("BASIC INFORMATION", state["company_info"]),
        ("BUSINESS MODEL", state["business_model"]),
        ("REVENUE SOURCES", state["revenue_sources"]),
        ("FINANCIAL ANALYSIS", state["financial_analysis"]),
        ("GROWTH TRIGGERS", state["growth_triggers"]),
        ("CAPEX AND ORDER BOOK ANALYSIS", state["capex_analysis"]),
        ("MARKET POSITION AND TAILWINDS", state["market_position"]),
        ("RISK ANALYSIS", state["risk_analysis"]),
        ("INVESTMENT RECOMMENDATION", state["investment_recommendation"])
    ]
    
    final_report = "\n\n".join(f"New section\n{content}" for title, content in sections)
    logger.info("✅ Final report compilation complete")
    return {"final_report": final_report, "messages": [AIMessage(content=final_report)]}

def route_data_collector(state):
    """Custom routing function for data collector"""
    # Check if the last message is from the LLM and contains tool calls
    last_message = state["messages"][-1]
    
    # If there are tool calls, route to tools
    if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls'):
        logger.info("🔄 Routing to tools")
        return "tools"
    
    # If we have collected data in context, route to analysis
    if state.get("context", {}).get("collected_data"):
        logger.info("✅ Data collection complete, routing to analysis")
        return "info_analysis"
    
    # If still collecting data but no tool calls, continue collecting
    logger.info("🔄 Continuing data collection")
    return "tools"

### Graph Construction
builder = StateGraph(ResearchState)

# Add nodes
builder.add_node("data_collector", data_collector)
builder.add_node("tools", ToolNode(tools))
builder.add_node("info_analysis", company_info_analyst)
builder.add_node("model_analysis", business_model_analyst)
builder.add_node("revenue_analysis", revenue_analyst)
builder.add_node("financial_analysis_node", financial_analyst)
builder.add_node("growth_analysis", growth_analyst)
builder.add_node("capex_analysis_node", capex_analyst)
builder.add_node("market_analysis", market_analyst)
builder.add_node("risk_analysis_node", risk_analyst)
builder.add_node("investment_analysis", investment_analyst)
builder.add_node("report_compilation", report_compiler)

# Add edges
builder.add_edge(START, "data_collector")
builder.add_conditional_edges(
    "data_collector",
    route_data_collector,  # Use our custom routing function
    {
        "tools": "tools",
        "info_analysis": "info_analysis"
    }
)
builder.add_edge("tools", "data_collector")

# Analysis flow
builder.add_edge("info_analysis", "model_analysis")
builder.add_edge("model_analysis", "revenue_analysis")
builder.add_edge("revenue_analysis", "financial_analysis_node")
builder.add_edge("financial_analysis_node", "growth_analysis")
builder.add_edge("growth_analysis", "capex_analysis_node")
builder.add_edge("capex_analysis_node", "market_analysis")
builder.add_edge("market_analysis", "risk_analysis_node")
builder.add_edge("risk_analysis_node", "investment_analysis")
builder.add_edge("investment_analysis", "report_compilation")
builder.add_edge("report_compilation", END)

# Compile graph
graph = builder.compile()
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

# Example usage
if __name__ == "__main__":
    result = graph.invoke({
        "messages": [
            HumanMessage(content="""Research time technoplast using these sources:
            You can parse this pdf https://www.timetechnoplast.com/wp-content/uploads/2024/12/q3-transcript.pdf
            You can crawl this website https://www.screener.in/company/TIMETECHNO/consolidated/
            Create a detailed report on Time technoplast.""")
        ]
    })
    # print(result)
    for m in result['messages']:
        m.pretty_print()
