from typing import List, Dict
from groq import Groq

from app.logger.log import logger
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

            prompt = f"""
You are an AI assistant for FinSolve Technologies. 
User Role: {user_role}

Based on the following context documents, answer the user's question.
Always cite the source documents in your response.

Context:
{context}

Question: {query}

Answer:
"""

            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant for FinSolve Technologies."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.3,
                max_tokens=1000
            )

            final_answer = response.choices[0].message.content
            logger.info("✅ Response generated successfully.")
            return final_answer

        except Exception as e:
            logger.exception("❌ Failed to generate LLM response.")
            raise LLMResponseError(query) from e

    def _prepare_context(self, documents: List[Dict]) -> str:
        """
        🧾 Format and concatenate context documents.

        Args:
            documents (List[Dict]): List of document chunks with content and metadata.

        Returns:
            str: Formatted string containing all context documents.
        """
        try:
            context_parts = []
            for doc in documents:
                source = doc.get('metadata', {}).get('source', 'Unknown')
                content = doc.get('content', '')
                context_parts.append(f"Source: {source}\nContent: {content}\n")

            combined_context = "\n---\n".join(context_parts)
            logger.debug(f"📚 Prepared context with {len(context_parts)} documents.")
            return combined_context

        except Exception as e:
            logger.exception("❌ Failed while preparing context documents.")
            raise ContextPreparationError() from e
