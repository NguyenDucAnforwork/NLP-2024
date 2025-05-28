"""
LLM-based generator implementation for the law document system.
"""
import os
from typing import List, Dict, Any, Optional, Tuple

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.gemini import Gemini
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank

from .base_generator import BaseGenerator
from .response_synthesizer import create_response_synthesizer


class Route(BaseModel):
    binary_score: str

class GradeDocuments(BaseModel):
    binary_scores: List[str]

class GradeUsefulness(BaseModel):
    score: int
    explanation: str

class RephraseQuery(BaseModel):
    new_query: str


class LLMGenerator(BaseGenerator):
    """
    Generator that uses a language model to generate responses.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        response_mode: str = "compact"
    ):
        """
        Initialize the LLM generator.
        
        Args:
            model_name: Name of the language model to use
            temperature: Temperature for generation
            api_key: API key for the language model (defaults to environment variable)
            response_mode: Mode for response synthesis
        """
        super().__init__(temperature=temperature)
        
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Initialize LLM
        self.llm = GoogleGenAI(api_key=self.api_key, model=model_name, temperature=temperature)
        self.logger.info(f"Initialized Gemini LLM with model {model_name}")
        
        # Response mode
        self.response_mode = response_mode
    
    def generate_response(self, query_str: str, nodes: List[NodeWithScore]|None, cutoff=0.0, rerank=False, verbose=False) -> str:
        """Generate a response based on the query and retrieved nodes.
        Args:
            query_str: Query string
            nodes: List of retrieved nodes, `None` for direct generation
            cutoff: Similarity cutoff for postprocessing
            rerank: Whether to apply reranking to the nodes
            verbose: Whether to log detailed information
        Returns:
            Generated response
        """
        if verbose:
            self.logger.info(f"Generating response for query: {query_str}")
        
        if not nodes:
            self.logger.info("No nodes provided, generating response directly")
            response = self.llm.complete(query_str)
            return response.text.strip()
        
        # Apply similarity postprocessing to get most relevant results
        postprocessor_cutoff = SimilarityPostprocessor(similarity_cutoff=cutoff)
        filtered_nodes = postprocessor_cutoff.postprocess_nodes(nodes)

        # If using reranking, apply it to the filtered nodes
        if rerank:
            postprocessor_rerank = SentenceTransformerRerank(
                model="./data/reranker/bge-reranker-v2-m3", 
                top_n=5
            )
            filtered_nodes = postprocessor_rerank.postprocess_nodes(filtered_nodes)

        if verbose:
            self.logger.info(f"After postprocessing: {len(filtered_nodes)} nodes")
        
        response_synthesizer = create_response_synthesizer(
            llm=self.llm,
            response_mode=self.response_mode,
        )
        
        # Generate response
        response = response_synthesizer.synthesize(query_str, filtered_nodes, verbose=verbose)
        
        return response
    

    def route_question(self, query_str: str) -> str:
        """Route the question to determine if information retrieval is needed.
        Args:
            query_str: User question string
        Returns:
            Binary score ('yes' or 'no') indicating if information retrieval is needed
        """
        prompt = PromptTemplate(
"You are an expert at routing questions to the right source of information. \n\n\
If the given question requires further information of the legal domain, mark it as information retrieval needed. \n\n\
Give a binary score 'yes' or 'no' to indicate whether information retrieval is needed. \n\n\
User question: {query}"
        )
        response_obj = self.llm.structured_predict(Route, prompt, query=query_str)
        if response_obj.binary_score not in ["yes", "no"]:
            return "yes"
        return response_obj.binary_score
    

    def rephrase_query(self, query_str: str, arch="standard") -> str:
        """Rephrase the user question for better search results.
        Args:
            query_str: User question string
            arch: Architecture type for rephrasing
                - "standard" for general rephrasing
                - "hyde" for hypothetical answer generation as phrasing
        Returns:
            Rephrased query string
        """
        prompt_standard = PromptTemplate(
"You are an expert at rephrasing questions to make them search-friendly. \n\n\
Rephrase the given user question into a search-friendly version, using synonyms and related terms. Prioritize technical terms from the legal domain. \n\n\
Use Vietnamese. \n\n\
User question: {query}"
        )
        prompt_hyde = PromptTemplate(
"Write a hypothetical paragraph answering the user question. \n\n\
Prioritize technical terms from the legal domain. \n\n\" \
Use Vietnamese. \n\n\
User question: {query}"
        )
        prompt = prompt_standard if arch == "standard" else prompt_hyde
        response_obj = self.llm.structured_predict(RephraseQuery, prompt, query=query_str)
        return response_obj.new_query.strip() if response_obj.new_query else query_str.strip()


    def grade_documents_relevance(self, query_str: str, nodes: List[NodeWithScore]) -> List[str]:
        """Grade the relevance of retrieved documents to the user question.
        Args:
            query_str: User question string
            nodes: List of retrieved nodes
        Returns:
            List of binary scores ('yes' or 'no') indicating relevance of each document
        """
        prompt = PromptTemplate(
"You are a grader assessing relevance of a list of retrieved documents to a user question. \n\n\
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n\n\
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n\n\
Give a list of binary score 'yes' or 'no' score to indicate whether the documents are relevant to the question. \n\n\
Retrieved documents: \n\n{documents} \n\n\
User question: {query}"
        )
        document_list = [node.node.get_content() for node in nodes]
        response_obj = self.llm.structured_predict(GradeDocuments, prompt, documents=" \n\n".join(document_list), query=query_str)
        if not isinstance(response_obj.binary_scores, list) or not all(score in ["yes", "no"] for score in response_obj.binary_scores):
            return ["yes"] * len(nodes)
        return response_obj.binary_scores
    

    def grade_answer_usefulness(self, query_str: str, answer: str) -> Tuple[int, str]:
        """Grade the usefulness of an answer to the user question.
        Args:
            query_str: User question string
            answer: Answer string
        Returns:
            Tuple containing score (1-5) and explanation
        """
        prompt = PromptTemplate(
"You are a grader assessing the usefulness and helpfulness of an answer to a user question. \n\n\
If the given answer addresses all aspects of the user question, grade it high in usefulness. \n\n\
Give a score from 1 to 5 to indicate whether the answer can address the question, and an explanation for that decision in a single sentence. \n\n\" \
Use Vietnamese. \n\n\
Answer: {answer} \n\n\
User question: {query}"
        )
        response_obj = self.llm.structured_predict(
            GradeUsefulness, prompt, answer=answer, query=query_str
        )
        if not hasattr(response_obj, 'score') or not isinstance(response_obj.score, int):
            return 1, "No score provided"
        score = max(1, min(5, response_obj.score))
        explanation = response_obj.explanation if hasattr(response_obj, 'explanation') else "No explanation provided"
        
        return score, explanation


if __name__ == "__main__":
    temp = PromptTemplate("hello, how are you?")
    print(temp)