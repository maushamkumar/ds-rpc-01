from typing import List, Dict
from groq import Groq
from app.logger.log import logging

logger = logging.getLogger(__name__)
from app.exception.exception_handler import (
    AppException,
    LLMResponseError,
    ContextPreparationError
)

class ResponseGenerator:
    def __init__(self, api_key: str):
        """
        🤖 Initialize the ResponseGenerator with Groq API client.
        
        Args:
            api_key (str): API key to authenticate with Groq service.
        """
        try:
            logger.info("🔐 Initializing Groq client for ResponseGenerator.")
            self.client = Groq(api_key=api_key)
            logger.info("✅ Groq client initialized successfully.")
        except Exception as e:
            logger.exception("❌ Failed to initialize Groq client.")
            raise AppException(e) from e
    
    def _get_role_specific_instructions(self, user_role: str) -> str:
        """
        📋 Get role-specific instructions for response generation.
        
        Args:
            user_role (str): Role of the user
            
        Returns:
            str: Role-specific instructions
        """
        role_instructions = {
            "Finance": """
            You are assisting a Finance team member. Focus on:
            - Financial metrics, KPIs, and performance indicators
            - Budget analysis, cost optimization, and expense tracking
            - Revenue reports, profit margins, and financial forecasting
            - Compliance, audit requirements, and regulatory matters
            - Investment analysis and financial risk assessment
            Provide precise numerical data when available and highlight financial implications.
            """,
            
            "Marketing": """
            You are assisting a Marketing team member. Focus on:
            - Campaign performance metrics (CTR, conversion rates, ROI)
            - Customer acquisition costs and customer lifetime value
            - Market trends, competitor analysis, and positioning
            - Brand awareness, engagement metrics, and social media performance
            - Lead generation, sales funnel optimization, and customer feedback
            Present data in actionable insights for marketing strategy decisions.
            """,
            
            "HR": """
            You are assisting an HR team member. Focus on:
            - Employee performance, attendance, and productivity metrics
            - Recruitment, onboarding, and retention statistics
            - Payroll, benefits, and compensation analysis
            - Training programs, skill development, and career progression
            - Workplace culture, employee satisfaction, and policy compliance
            Maintain confidentiality and provide insights for workforce optimization.
            """,
            
            "Engineering": """
            You are assisting an Engineering team member. Focus on:
            - Technical architecture, system design, and infrastructure
            - Development processes, code quality, and deployment metrics
            - Performance optimization, scalability, and security measures
            - Project timelines, resource allocation, and technical debt
            - Innovation opportunities, technology stack decisions, and best practices
            Provide technical depth and actionable engineering recommendations.
            """,
            
            "C-Level": """
            You are assisting a C-Level Executive. Provide:
            - High-level strategic insights across all departments
            - Cross-functional performance analysis and correlations
            - Executive summary format with key takeaways
            - Business impact assessment and growth opportunities
            - Risk analysis and strategic recommendations
            - Comprehensive view connecting financial, operational, and market data
            Focus on decision-making support and strategic planning.
            """,
            
            "Employee": """
            You are assisting a general employee. Focus on:
            - Company policies, procedures, and general information
            - Event announcements, updates, and FAQ responses
            - Basic organizational structure and contact information
            - General benefits, facilities, and resource availability
            - Non-sensitive company news and public information
            Keep responses helpful but limit access to sensitive departmental data.
            """
        }
        
        return role_instructions.get(user_role, role_instructions["Employee"])
    
    def _get_response_format_guidelines(self, user_role: str) -> str:
        """
        📝 Get response format guidelines based on user role.
        
        Args:
            user_role (str): Role of the user
            
        Returns:
            str: Format guidelines
        """
        if user_role == "C-Level":
            return """
            Format your response as:
            🎯 **Executive Summary**: Brief overview (2-3 sentences)
            📊 **Key Insights**: Main findings with metrics
            💡 **Strategic Implications**: Business impact and opportunities
            📋 **Recommendations**: Actionable next steps
            📚 **Sources**: Document references
            """
        elif user_role in ["Finance", "Marketing", "HR", "Engineering"]:
            return """
            Format your response as:
            📋 **Analysis**: Direct answer to the query
            📊 **Data Insights**: Relevant metrics and trends
            🔍 **Details**: Supporting information and context
            💡 **Recommendations**: Actionable suggestions for your department
            📚 **Sources**: Referenced documents
            """
        else:  # Employee level
            return """
            Format your response as:
            ✅ **Answer**: Clear, direct response
            ℹ️ **Additional Info**: Helpful context or related information
            📚 **Reference**: Source documents (if applicable)
            """

    def generate_response(self, query: str, context_documents: List[Dict], user_role: str) -> str:
        """
        🧠 Generate a response using Groq LLM based on user query and context.
        
        Args:
            query (str): The user's question.
            context_documents (List[Dict]): Relevant document chunks with metadata.
            user_role (str): Role of the user asking the question.
        
        Returns:
            str: AI-generated answer.
        """
        try:
            logger.info(f"📝 Generating response for user role: {user_role}, Query: {query[:50]}...")
            context = self._prepare_context(context_documents)
            role_instructions = self._get_role_specific_instructions(user_role)
            format_guidelines = self._get_response_format_guidelines(user_role)
            
            # Enhanced system prompt
            system_prompt = f"""You are FinSolve Assistant, an intelligent AI chatbot for FinSolve Technologies, a leading FinTech company providing innovative financial solutions.

ROLE-BASED ACCESS CONTROL:
- Current user role: {user_role}
- Provide responses tailored to this role's responsibilities and access level
- Maintain data security by respecting role boundaries

RESPONSE GUIDELINES:
{role_instructions}

{format_guidelines}

QUALITY STANDARDS:
- Be precise, professional, and actionable
- Use specific metrics and data when available
- Cite sources clearly and accurately
- If data is insufficient, acknowledge limitations
- Provide context-aware insights relevant to FinSolve's business
- Maintain confidentiality and data security protocols"""

            user_prompt = f"""
CONTEXT DOCUMENTS:
{context}

USER QUERY: {query}

Please provide a comprehensive response based on the context documents above, following the role-specific guidelines for {user_role} role. Ensure your answer is:
1. Directly relevant to the query
2. Appropriately detailed for the user's role
3. Backed by the provided context
4. Properly formatted according to the guidelines
5. Includes proper source citations

If the context doesn't contain sufficient information to answer the query, clearly state this and provide what relevant information is available."""

            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.2,  # Reduced for more consistent responses
                max_tokens=1500,   # Increased for detailed responses
                top_p=0.9
            )
            
            final_answer = response.choices[0].message.content
            logger.info("✅ Response generated successfully.")
            return final_answer
            
        except Exception as e:
            logger.exception("❌ Failed to generate LLM response.")
            raise LLMResponseError(query) from e

    def _prepare_context(self, documents: List[Dict]) -> str:
        """
        🧾 Format and concatenate context documents with enhanced metadata.
        
        Args:
            documents (List[Dict]): List of document chunks with content and metadata.
        
        Returns:
            str: Formatted string containing all context documents.
        """
        try:
            if not documents:
                return "No relevant documents found."
            
            context_parts = []
            for i, doc in enumerate(documents, 1):
                metadata = doc.get('metadata', {})
                source = metadata.get('source', 'Unknown Source')
                content = doc.get('content', '').strip()
                
                # Add additional metadata if available
                extra_info = []
                if 'department' in metadata:
                    extra_info.append(f"Department: {metadata['department']}")
                if 'document_type' in metadata:
                    extra_info.append(f"Type: {metadata['document_type']}")
                if 'date' in metadata:
                    extra_info.append(f"Date: {metadata['date']}")
                
                extra_info_str = " | ".join(extra_info)
                if extra_info_str:
                    source_line = f"[Document {i}] Source: {source} ({extra_info_str})"
                else:
                    source_line = f"[Document {i}] Source: {source}"
                
                context_parts.append(f"{source_line}\nContent: {content}")
            
            combined_context = "\n\n" + "="*50 + "\n\n".join(context_parts)
            logger.debug(f"📚 Prepared context with {len(context_parts)} documents.")
            return combined_context
            
        except Exception as e:
            logger.exception("❌ Failed while preparing context documents.")
            raise ContextPreparationError() from e