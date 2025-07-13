import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, save_tool, wiki_tool
import json

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def setup_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a research assistant that will help generate a research paper.
        Answer the user query and use necessary tools. 
        Wrap the output in this format and provide no other text\n{format_instructions}
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    tools = [search_tool, save_tool, wiki_tool]
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor, parser

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔬",
        layout="wide"
    )
    
    # Header
    st.title("🔬 AI Research Assistant")
    st.markdown("*Your intelligent research companion powered by Google Gemini*")
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Features")
        st.markdown("""
        - 🔍 **Web Search**: Latest information from the internet
        - 📖 **Wikipedia**: Comprehensive background knowledge
        - 💾 **Save Results**: Export your research findings
        - 📊 **Structured Output**: Clean, organized summaries
        """)
        
        st.header("🎯 Tips")
        st.markdown("""
        - Be specific with your research questions
        - Ask for comparisons or analysis
        - Request recent information for current topics
        """)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Research Query")
        query = st.text_area(
            "What would you like to research?",
            placeholder="e.g., 'Latest developments in artificial intelligence', 'Climate change effects on polar bears', etc.",
            height=100
        )
        
        if st.button("🚀 Start Research", type="primary"):
            if query:
                agent_executor, parser = setup_agent()
                
                with st.spinner("🔍 Conducting research..."):
                    try:
                        raw_response = agent_executor.invoke({"query": query})
                        
                        # Parse response
                        output = raw_response.get("output")
                        if isinstance(output, str):
                            structured_response = parser.parse(output)
                        else:
                            structured_response = parser.parse(output.get("text", ""))
                        
                        # Display results
                        st.success("✅ Research completed!")
                        
                        # Results section
                        st.header("📋 Research Results")
                        
                        # Topic
                        st.subheader(f"📌 Topic: {structured_response.topic}")
                        
                        # Summary
                        st.subheader("📄 Summary")
                        st.markdown(structured_response.summary)
                        
                        # Sources
                        if structured_response.sources:
                            st.subheader("📚 Sources")
                            for i, source in enumerate(structured_response.sources, 1):
                                st.markdown(f"{i}. {source}")
                        
                        # Tools used
                        if structured_response.tools_used:
                            st.subheader("🔧 Tools Used")
                            st.markdown(", ".join(structured_response.tools_used))
                        
                        # Download option
                        research_data = {
                            "topic": structured_response.topic,
                            "summary": structured_response.summary,
                            "sources": structured_response.sources,
                            "tools_used": structured_response.tools_used
                        }
                        
                        st.download_button(
                            label="📥 Download Research Data",
                            data=json.dumps(research_data, indent=2),
                            file_name=f"research_{structured_response.topic.replace(' ', '_')}.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.expander("Raw Response").write(raw_response)
            else:
                st.warning("⚠️ Please enter a research query.")
    
    with col2:
        st.subheader("📊 Research Stats")
        
        # Initialize session state
        if 'research_history' not in st.session_state:
            st.session_state.research_history = []
        
        # Display stats
        st.metric("Total Researches", len(st.session_state.research_history))
        
        # Recent queries
        if st.session_state.research_history:
            st.subheader("🕒 Recent Queries")
            for query in st.session_state.research_history[-5:]:
                st.markdown(f"• {query}")

if __name__ == "__main__":
    main()