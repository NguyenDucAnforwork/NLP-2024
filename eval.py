"""
Evaluation script for Vietnamese legal document QA system using RAGAS metrics.
"""
import os
import sys
import json
import pickle
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from datetime import datetime
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import RAGAS for evaluation
from ragas.metrics import (
    context_precision,
    context_recall, 
    faithfulness,
    answer_relevancy
)
from ragas import evaluate
from datasets import Dataset

# Import our custom modules
from src.backend.vector_indexes import ParentVectorIndex, ChildVectorIndex
from src.backend.generators import LLMGenerator

from llama_index.core.retrievers import AutoMergingRetriever, VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.evaluation import EvaluationResult
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
QA_DATA_PATH = os.path.join("qa_data", "qa_pairs_full.json")
EMBED_MODEL_PATH = "BAAI/bge-m3"
LOCAL_EMBED_MODEL_PATH = "./data/embeddings/Vietnamese_Embedding"
PARENT_COLLECTION = "vietnamese_legal_parent"
LOCAL_PARENT_COLLECTION = "./data/nodes/hienphap_parent.pkl"
CHILD_COLLECTION = "vietnamese_legal_child"
LOCAL_CHILD_COLLECTION = "./data/nodes/hienphap_child.pkl"
MODEL_NAME = "gemini-2.0-flash"
TEMPERATURE = 0.1  # Lower temperature for evaluation
num_generators = 8 # More generators to avoid rate limit
OUTPUT_DIR = "evaluation_results"
EVALUATION_SAMPLE_SIZE = 4000  # Number of QA pairs to evaluate
os.environ["IS_TESTING"] = "1"  # Avoid using LLM for retrieval

class RAGEvaluator:
    """Class to evaluate RAG pipeline using RAGAS metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        logger.info("Initializing RAG Evaluator...")
        self.parent_index = None
        self.child_index = None
        self.parent_nodes = None
        self.child_nodes = None
        self.storage_context = None
        self.embed_model = None

        self.generators = []
        self.num_generators = 0
        self.rest_between_gens = 0

        self.retrievers = {}

        self.qa_pairs = None
        self.evaluation_results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        

    def load_qa_data(self) -> None:
        """Load QA pairs from JSON file."""
        logger.info(f"Loading QA pairs from {QA_DATA_PATH}...")
        try:
            with open(QA_DATA_PATH, 'r', encoding='utf-8') as f:
                self.qa_pairs = json.load(f)
            
            # Sample a subset for evaluation
            if len(self.qa_pairs) > EVALUATION_SAMPLE_SIZE:
                np.random.seed(42)  # For reproducibility
                self.qa_pairs = np.random.choice(
                    self.qa_pairs, 
                    EVALUATION_SAMPLE_SIZE, 
                    replace=False
                ).tolist()
                
            logger.info(f"Loaded {len(self.qa_pairs)} QA pairs for evaluation")
        except Exception as e:
            logger.error(f"Error loading QA pairs: {e}")
            raise
    

    def load_indexes(self, load_from='cloud') -> None:
        """Load parent and child indexes from a location.
        Args:
            load_from: 'cloud' to load from Qdrant Cloud, 'local' for local files
        """
        if load_from == 'cloud':
            logger.info("Loading indexes from Qdrant Cloud...")
            try:
                # Initialize parent index
                parent_idx = ParentVectorIndex(
                    collection_name=PARENT_COLLECTION,
                    embed_model_path=EMBED_MODEL_PATH
                )
                self.parent_index, self.parent_nodes = parent_idx.load_index_from_cloud()
                logger.info(f"Loaded parent index with {len(self.parent_nodes)} nodes")
                
                # Initialize child index
                child_idx = ChildVectorIndex(
                    collection_name=CHILD_COLLECTION,
                    embed_model_path=EMBED_MODEL_PATH
                )
                self.child_index, self.child_nodes = child_idx.load_index_from_cloud()
                logger.info(f"Loaded child index with {len(self.child_nodes)} nodes")
                
                # Create storage context and add all nodes
                self._setup_storage_context()
            except Exception as e:
                logger.error(f"Error loading indexes: {e}")
                raise
        elif load_from == 'local':
            logger.info("Loading indexes from local files...")
            try:
                # Load parent nodes
                with open(LOCAL_PARENT_COLLECTION, 'rb') as f:
                    self.parent_nodes = pickle.load(f)
                logger.info(f"Loaded {len(self.parent_nodes)} parent nodes")
                
                # Load child nodes
                with open(LOCAL_CHILD_COLLECTION, 'rb') as f:
                    self.child_nodes = pickle.load(f)
                logger.info(f"Loaded {len(self.child_nodes)} child nodes")

                self.parent_index = VectorStoreIndex(self.parent_nodes, embed_model=self.embed_model)
                self.child_index = VectorStoreIndex(self.child_nodes, embed_model=self.embed_model)
                
                # Create storage context and add all nodes
                self._setup_storage_context()
            except Exception as e:
                logger.error(f"Error loading indexes: {e}")
                raise
        else:
            logger.error("Invalid load_from option, must be 'cloud' or 'local'")
            raise ValueError("Invalid load_from option, must be 'cloud' or 'local'")


    def _setup_storage_context(self) -> None:
        """Set up storage context with parent-child relationships."""
        logger.info("Setting up storage context with parent-child relationships...")
        docstore = SimpleDocumentStore()
        docstore.add_documents(self.parent_nodes)
        
        # Add child nodes and establish parent-child relationships
        for child in self.child_nodes:
            if 'parent_id' in child.metadata:
                parent_id = child.metadata['parent_id']
                try:
                    parent_node = docstore.get_document(parent_id)
                    if isinstance(parent_node, BaseNode):
                        child.parent_node = {"node_id": parent_id}
                except:
                    logger.warning(f"Could not find parent node with ID {parent_id}")
        
        docstore.add_documents(self.child_nodes)
        
        # Create storage context with the docstore
        self.storage_context = StorageContext.from_defaults(docstore=docstore)
        

    def initialize_generators(self) -> None:
        """Initialize the LLM generators."""
        try:
            self.generators = []
            for i in range(num_generators):
                self.generators.append(
                    LLMGenerator(
                        model_name=MODEL_NAME,
                        temperature=TEMPERATURE,
                        api_key=os.getenv(f"GEMINI_API_KEY_{i}"),
                    )
                )
                logger.info(f"Initializing generator with model: {MODEL_NAME} and key GEMINI_API_KEY_{i}")
            self.rest_between_gens = (60 / (15 * num_generators))  # Rest time between generators to avoid rate limit
            self.num_generators = len(self.generators)
        except Exception as e:
            logger.error(f"Error initializing generator: {e}")
            raise
    

    def initialize_embed_model(self) -> None:
        """Initialize the embedding model from local path."""
        self.embed_model = HuggingFaceEmbedding(model_name=LOCAL_EMBED_MODEL_PATH)


    def setup_retrievers(self, llm=None) -> None:
        """Set up different retrieval methods for evaluation using standard LlamaIndex implementations.
        Args:
            llm: Optional LLM instance for retrieval methods that require it
        """
        logger.info("Setting up different retrieval methods...")
        try:            
            # 1. Basic vector retriever from child index
            vector_retriever = self.child_index.as_retriever(similarity_top_k=5)
            self.retrievers["vector"] = vector_retriever
            logger.info("Successfully initialized vector retriever")
            
            # 2. BM25 retriever (purely lexical search)
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.storage_context.docstore,
                similarity_top_k=5
            )
            self.retrievers["bm25"] = bm25_retriever
            logger.info("Successfully initialized BM25 retriever")
            
            # 3. Hybrid retriever using reciprocal rank fusion
            try:
                # Create explicit vector retriever
                vector_retriever_for_fusion = VectorIndexRetriever(
                    index=self.child_index,
                    similarity_top_k=5
                )
                # Create fusion retriever (combines vector and keyword search)
                fusion_retriever = QueryFusionRetriever(
                    llm=llm, num_queries=4 if llm else 1,
                    retrievers=[vector_retriever_for_fusion, bm25_retriever],
                    similarity_top_k=5,
                    mode="reciprocal_rerank",
                )
                self.retrievers["hybrid_fusion"] = fusion_retriever
                logger.info("Successfully initialized hybrid fusion retriever")
            except Exception as e:
                logger.warning(f"Could not initialize hybrid fusion retriever: {e}")

        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
            raise


    def save_results(self) -> None:
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"rag_evaluation_{timestamp}.json")
        
        logger.info(f"Saving evaluation results to {output_file}...")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_results, f, indent=4, ensure_ascii=False)
            
            # Also create a summary markdown file
            summary_file = os.path.join(OUTPUT_DIR, f"rag_evaluation_summary_{timestamp}.md")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("# RAG Evaluation Summary\n\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Sample size: {len(self.qa_pairs)} QA pairs\n\n")
                
                f.write("## Results by Retriever\n\n")
                
                # Create markdown table
                f.write("| Retriever | Context Precision | Context Recall | Faithfulness | Answer Relevancy |\n")
                f.write("|-----------|------------------|----------------|--------------|------------------|\n")
                
                for retriever_name, metrics in self.evaluation_results.items():
                    f.write(f"| {retriever_name} | {metrics.get('context_precision', 0):.4f} | {metrics.get('context_recall', 0):.4f} | {metrics.get('faithfulness', 0):.4f} | {metrics.get('answer_relevancy', 0):.4f} |\n")
                
                f.write("\n\n## Explanation of Metrics\n\n")
                f.write("- **Context Precision**: Measures how much of the retrieved context is relevant\n")
                f.write("- **Context Recall**: Measures how much of the required information is retrieved\n")
                f.write("- **Faithfulness**: Measures how factually consistent the answer is with the retrieved context\n")
                f.write("- **Answer Relevancy**: Measures how relevant the answer is to the question\n")
                
            logger.info(f"Saved evaluation summary to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")


    def retrieve_all_and_save(self) -> None:
        """
        Retrieve contexts for all questions using all retrievers and save results to disk.
        This separates retrieval from generation to optimize for API rate limits.
        """
        logger.info("Starting retrieval for all questions with all retrievers...")
        
        # Dictionary to store retrieval results for all methods
        all_retrieval_results = {}
        
        for retriever_name, retriever in self.retrievers.items():
            logger.info(f"Retrieving contexts using {retriever_name} retriever...")
            retrieval_results = {}
            
            for idx, qa_pair in tqdm(enumerate(self.qa_pairs), desc=f"Retrieving with {retriever_name}", total=len(self.qa_pairs)):
                question = qa_pair.get("question", "")
                if not question:
                    continue
                    
                try:
                    retrieved_nodes = retriever.retrieve(question)
                    
                    # Store the retrieval result
                    retrieval_results[str(idx)] = {
                        "question": question,
                        "retrieved_nodes": retrieved_nodes,
                        "ground_truth": qa_pair.get("answer", ""),
                    }
                    
                except Exception as e:
                    logger.error(f"Error retrieving for question #{idx} with {retriever_name}: {e}")
            
            all_retrieval_results[retriever_name] = retrieval_results
            
            # Save retrieval results for this method as pickle file
            retrieval_file = os.path.join(OUTPUT_DIR, f"retrieval_results_{retriever_name}.pkl")
            with open(retrieval_file, 'wb') as f:
                pickle.dump(retrieval_results, f)

            logger.info(f"Saved retrieval results for {retriever_name} to {retrieval_file}")
        
        # # Also save all retrieval results in one file for convenience
        # all_retrieval_file = os.path.join(OUTPUT_DIR, "all_retrieval_results.json")
        # with open(all_retrieval_file, 'w', encoding='utf-8') as f:
        #     json.dump({
        #         "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        #         "results": all_retrieval_results
        #     }, f, indent=4, ensure_ascii=False)
        
        # logger.info(f"Saved all retrieval results to {all_retrieval_file}")
        
        return all_retrieval_results
    

    def generate_answers_from_retrieval(self, retriever_name: str, retrieval_results: Dict = None) -> Dict:
        """
        Generate answers using previously retrieved contexts.
        
        Args:
            retriever_name: Name of the retriever whose results to use
            retrieval_results: Optional pre-loaded retrieval results
            
        Returns:
            Dictionary with question, contexts, ground truth and generated answers
        """
        logger.info(f"Generating answers for {retriever_name} retrieval results...")
        
        # Load retrieval results if not provided
        if retrieval_results is None:
            retrieval_file = os.path.join(OUTPUT_DIR, f"retrieval_results_{retriever_name}.pkl")
            try:
                with open(retrieval_file, 'rb') as f:
                    retrieval_results = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading retrieval results for {retriever_name}: {e}")
                return {}
        
        # Prepare data for RAGAS
        questions = []
        ground_truths = []
        contexts_list = []
        answers = []
        total_num_questions = len(retrieval_results)
        print(f"Total number of questions: {total_num_questions}")
        start_from = 0
        
        # Load previous run if exists and not done
        gen_file = os.path.join(OUTPUT_DIR, f"generation_results_{retriever_name}.json")
        if os.path.exists(gen_file):
            with open(gen_file, 'r', encoding='utf-8') as f:
                gen_data = json.load(f)
                prev_data_dict = gen_data["data"]
                if len(prev_data_dict["question"]) < total_num_questions:
                    start_from = len(prev_data_dict["question"])
                    questions = prev_data_dict["question"]
                    ground_truths = prev_data_dict["ground_truth"]
                    contexts_list = prev_data_dict["contexts"]
                    answers = prev_data_dict["answer"]
                    logger.info(f"Resuming from question index {start_from} for {retriever_name}")

        # For continuation in case previous run was interrupted
        if start_from > 0:
            retrieval_results = {k: v for k, v in retrieval_results.items() if int(k) >= start_from}
        
        # Process each question
        prematurely_done = False

        for idx, result in tqdm(retrieval_results.items(), desc=f"Generating answers for {retriever_name}",
                                total=total_num_questions, initial=start_from):
            question = result.get("question", "")
            ground_truth = result.get("ground_truth", "")

            retrieved_nodes = result.get("retrieved_nodes", [])
            # Handle different node formats and extract text
            contexts = []
            for node in retrieved_nodes:
                if hasattr(node, 'node') and hasattr(node.node, 'text'):
                    contexts.append(node.node.text)
                elif hasattr(node, 'text'):
                    contexts.append(node.text)
                elif isinstance(node, dict) and 'text' in node:
                    contexts.append(node['text'])
                else:
                    logger.warning(f"Unknown node format: {type(node)}")
            
            if not question:
                continue
                
            # Generate answer
            done = False
            retries = 0
            gen_idx = (int(idx) // 15) % self.num_generators
            generator = self.generators[gen_idx]

            while not done:
                if retries == 5:
                    if len(self.generators) > 1:  # Don't remove the last generator
                        self.generators.pop(gen_idx)
                        self.num_generators -= 1
                        gen_idx = (int(idx) // 15) % max(1, self.num_generators)
                        generator = self.generators[gen_idx]
                    else:
                        logger.error(f"All generators failed, you might need to wait tomorrow. Note down the idx of the failed question: {idx}")
                        prematurely_done = True
                        break
                    retries = 0
                try:
                    response = generator.generate_response(question, retrieved_nodes)
                    answer = response.response.strip()
                    # time.sleep(self.rest_between_gens)  # Respect rate limits
                    done = True
                except Exception as e:
                    logger.error(f"Error with generator #{gen_idx}: {e}")
                    time.sleep(5)
                    retries += 1
                    continue
            
            if prematurely_done:
                break

            # Add to our lists
            questions.append(question)
            ground_truths.append(ground_truth)
            contexts_list.append(contexts)
            answers.append(answer)
        
        # Create data dictionary
        data_dict = {
            "question": questions,
            "ground_truth": ground_truths,
            "answer": answers,
            "contexts": contexts_list
        }

        # Save generated data
        with open(gen_file, 'w', encoding='utf-8') as f:
            json.dump({
                "retriever": retriever_name,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "data": data_dict
            }, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Saved generation results for {retriever_name} to {gen_file}")
        
        return data_dict
    

    def evaluate_from_saved_data(self, retriever_name: str) -> Dict[str, float]:
        """
        Run RAGAS evaluation using saved generation data.
        
        Args:
            retriever_name: Name of the retriever to evaluate
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Running RAGAS evaluation for {retriever_name} from saved data...")
        
        try:
            # Load generation results
            gen_file = os.path.join(OUTPUT_DIR, f"generation_results_{retriever_name}.json")
            with open(gen_file, 'r', encoding='utf-8') as f:
                gen_data = json.load(f)
                data_dict = gen_data["data"]
            
            # Create dataset for RAGAS
            dataset = Dataset.from_dict(data_dict)
            
            # Run RAGAS evaluation
            result = evaluate(
                dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ]
            )
            
            # Save results to pandas DataFrame and to csv file
            df = result.to_pandas()
            csv_file = os.path.join(OUTPUT_DIR, f"ragas_results_{retriever_name}.csv")
            df.to_csv(csv_file, index=False)
            
            # Extract scores
            metrics = {
                "context_precision": result["context_precision"].mean(),
                "context_recall": result["context_recall"].mean(),
                "faithfulness": result["faithfulness"].mean(),
                "answer_relevancy": result["answer_relevancy"].mean(),
            }
            
            logger.info(f"RAGAS metrics for {retriever_name}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error running RAGAS evaluation for {retriever_name}: {e}")
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
            }
    

    def run_retrieval_phase(self) -> None:
        """Run the retrieval phase for all retrievers."""
        logger.info("Starting retrieval phase...")
        self.retrieve_all_and_save()
        logger.info("Retrieval phase completed!")


    def run_generation_phase(self, retriever_names=None) -> None:
        """
        Run the generation phase for selected retrievers.
        
        Args:
            retriever_names: List of retriever names to process, or None for all
        """
        if retriever_names is None:
            retriever_names = list(self.retrievers.keys())
            
        logger.info(f"Starting generation phase for retrievers: {retriever_names}...")
        
        for retriever_name in retriever_names:
            logger.info(f"Generating answers for {retriever_name}...")
            self.generate_answers_from_retrieval(retriever_name)
        
        logger.info("Generation phase completed!")
    

    def run_evaluation_phase(self, retriever_names=None) -> None:
        """
        Run the evaluation phase for selected retrievers.
        
        Args:
            retriever_names: List of retriever names to evaluate, or None for all
        """
        if retriever_names is None:
            retriever_names = list(self.retrievers.keys())
            
        logger.info(f"Starting evaluation phase for retrievers: {retriever_names}...")
        
        for retriever_name in retriever_names:
            logger.info(f"Evaluating {retriever_name}...")
            metrics = self.evaluate_from_saved_data(retriever_name)
            self.evaluation_results[retriever_name] = metrics
        
        # Save results
        self.save_results()
        logger.info("Evaluation phase completed!")


    def answer_query(self, query: str, retriever_type: str = "vector", cutoff: float = 0.0, 
                    use_routing: bool = False, use_rephrasing: bool = False, rephrase_arch: str = "standard",
                    use_grading: bool = False, grade_answer: bool = False) -> Tuple[str, Dict]:
        """
        Process a query and return an answer with sources using the complete RAG pipeline.
        
        Args:
            query: Question to answer
            retriever_type: Type of retriever to use (default is "vector")
            cutoff: Similarity cutoff for postprocessing (default is 0.0)
            use_routing: Whether to route the query first (default is False)
            use_rephrasing: Whether to rephrase the query (default is False)
            rephrase_arch: Architecture for rephrasing ("standard" or "hyde")
            use_grading: Whether to grade retrieved documents (default is False)
            grade_answer: Whether to grade the final answer usefulness (default is False)
            
        Returns:
            Tuple of:
                - result: Answer to the question
                - answer_metadata: Dictionary with metadata about the answer generation process
                    - retrieve: Routing decision if used, "yes" or "no"
                    - rephrase: Rephrased query if rephrasing was used
                    - rephrase_arch: Architecture used for rephrasing, "standard" or "hyde"
                    - retriever_type: Type of retriever used
                    - retrieved_nodes: List of retrieved nodes (documents) used for answer generation
                    - relevance: Relevance scores of retrieved documents if grading was used
                    - usefulness_score: Score of the final answer usefulness if grading was used
                    - usefulness_explanation: Explanation of the usefulness score if grading was used
        """
        if not query:
            return "Vui lòng nhập câu hỏi.", []
        
        try:
            answer_metadata = {}
            original_query = query
            
            # Use the first available generator
            if not self.generators:
                return "No generators available.", answer_metadata
            
            generator = self.generators[-1]
            
            # Step 1: Route the query (optional)
            if use_routing:
                logger.info("Step 1: Routing query...")
                routing_decision = generator.route_question(query)
                answer_metadata["retrieve"] = routing_decision
                
                if routing_decision.lower() == "no":
                    result = generator.generate_response(query_str=query, nodes=None, cutoff=cutoff)
                    return str(result), answer_metadata
            
            # Step 2: Rephrase the query (optional)
            processed_query = query
            if use_rephrasing:
                logger.info(f"Step 2: Rephrasing query using {rephrase_arch} architecture...")
                processed_query = generator.rephrase_query(query, rephrase_arch)
                answer_metadata["rephrase"] = processed_query
                answer_metadata["rephrase_arch"] = rephrase_arch
            
            # Step 3: Retrieve documents
            logger.info("Step 3: Retrieving documents...")
            retriever = self.retrievers.get(retriever_type)
            if not retriever:
                return f"Retriever type '{retriever_type}' not found.", []
            
            retrieved_nodes = retriever.retrieve(processed_query)
            
            # Step 4: Grade retrieved documents (optional)
            relevant_nodes = retrieved_nodes
            answer_metadata["retriever_type"] = retriever_type
            answer_metadata["retrieved_nodes"] = [node.node.text for node in retrieved_nodes]

            if use_grading:
                logger.info("Step 4: Grading document relevance...")
                scores = generator.grade_documents_relevance(processed_query, retrieved_nodes)
                
                # Filter only relevant documents
                relevant_nodes = []
                relevant_count = 0
                for node, score in zip(retrieved_nodes, scores):
                    if score.lower() == "yes":
                        relevant_nodes.append(node)
                        relevant_count += 1

                if not relevant_nodes:
                    result = generator.generate_response(query_str=processed_query, nodes=None, cutoff=cutoff)
                    return str(result), answer_metadata
                
                answer_metadata["relevance"] = scores
            
            # Step 5: Generate answer
            logger.info("Step 5: Generating response...")
            response = generator.generate_response(query_str=processed_query, nodes=relevant_nodes, cutoff=cutoff)
            
            # Step 6: Grade answer usefulness (optional)
            if grade_answer:
                logger.info("Step 6: Grading answer usefulness...")
                score, explanation = generator.grade_answer_usefulness(original_query, str(response))
                answer_metadata["usefulness_score"] = score
                answer_metadata["usefulness_explanation"] = explanation
            
            # Format source information
            source_info = []
            nodes_to_show = relevant_nodes if use_grading else retrieved_nodes
            for i, source in enumerate(nodes_to_show[:3], 1):  # Display top 3 sources
                metadata = source.node.metadata
                doc_id = metadata.get("document_id", "Unknown")
                article = metadata.get("article", "")
                chapter = metadata.get("chapter", "")
                
                source_text = source.node.text[:150] + "..." if len(source.node.text) > 150 else source.node.text
                
                source_str = f"Nguồn {i}: Văn bản: {doc_id}\n"
                if chapter:
                    source_str += f"Chương: {chapter}\n"
                if article:
                    source_str += f"Điều: {article}\n"
                source_str += f"Trích đoạn: {source_text}\n\n"
                source_info.append(source_str)
            
            return str(response), answer_metadata
        
        except Exception as e:
            error_msg = f"Lỗi khi tạo câu trả lời: {str(e)}"
            logger.error(error_msg)
            return error_msg, []


    def route_query(self, query: str) -> str:
        """
        Process a query and determine if information retrieval is needed.
        
        Args:
            query: Question to route
            
        Returns:
            String showing the routing decision
        """
        if not query:
            return "Vui lòng nhập câu hỏi."
        
        try:
            # Use the first available generator to route the question
            if not self.generators:
                return "No generators available."
                
            generator = self.generators[-1]
            
            # Route the question
            logger.info("Routing question...")
            routing_decision = generator.route_question(query)
            
            # Format the results
            result_lines = [
                f"Câu hỏi: {query}",
                "",
                "Kết quả định tuyến câu hỏi:",
                "=" * 50
            ]
            
            # Format routing decision
            if routing_decision.lower() == "yes":
                decision_status = "✅ CÓ - Cần truy xuất thông tin pháp luật"
                explanation = "Câu hỏi này có liên quan đến lĩnh vực pháp luật và cần truy xuất thông tin từ cơ sở dữ liệu."
            else:
                decision_status = "❌ KHÔNG - Không cần truy xuất thông tin pháp luật"
                explanation = "Câu hỏi này không liên quan đến lĩnh vực pháp luật hoặc có thể trả lời mà không cần truy xuất thông tin."
            
            result_lines.extend([
                f"Quyết định: {decision_status}",
                f"Điểm số: {routing_decision}",
                "",
                "Giải thích:",
                explanation,
                "",
                "Hướng dẫn:",
                "- 'yes': Hệ thống sẽ tìm kiếm trong cơ sở dữ liệu văn bản pháp luật",
                "- 'no': Hệ thống có thể trả lời trực tiếp hoặc từ chối câu hỏi"
            ])
            
            return "\n".join(result_lines)
        
        except Exception as e:
            error_msg = f"Lỗi khi định tuyến câu hỏi: {str(e)}"
            logger.error(error_msg)
            return error_msg
    

    def rephrase_query(self, query: str, arch: str = "standard") -> str:
        """
        Process a query and rephrase it for better search results.
        
        Args:
            query: Question to rephrase
            arch: Architecture type for rephrasing ("standard" or "hyde")
            
        Returns:
            String showing the original and rephrased query
        """
        if not query:
            return "Vui lòng nhập câu hỏi."
        
        try:
            # Use the first available generator to rephrase the question
            if not self.generators:
                return "No generators available."
                
            generator = self.generators[-1]
            
            # Rephrase the question
            logger.info(f"Rephrasing question using {arch} architecture...")
            rephrased_query = generator.rephrase_query(query, arch)
            
            # Format the results
            result_lines = [
                f"Câu hỏi gốc: {query}",
                "",
                "Kết quả chuyển đổi câu hỏi:",
                "=" * 50
            ]
            
            # Format architecture description
            if arch == "standard":
                arch_description = "Chuyển đổi thành câu hỏi thân thiện với tìm kiếm"
                explanation = "Sử dụng từ đồng nghĩa và các thuật ngữ liên quan để cải thiện kết quả tìm kiếm."
            else:  # hyde
                arch_description = "Tạo đoạn văn giả định trả lời câu hỏi"
                explanation = "Tạo ra một đoạn văn giả định có thể trả lời câu hỏi để cải thiện tìm kiếm ngữ nghĩa."
            
            result_lines.extend([
                f"Kiến trúc: {arch.upper()} - {arch_description}",
                "",
                "Giải thích:",
                explanation,
                "",
                "Câu hỏi sau khi chuyển đổi:",
                f'"{rephrased_query}"',
                "",
                "So sánh:",
                f"- Gốc: {len(query)} ký tự",
                f"- Sau chuyển đổi: {len(rephrased_query)} ký tự",
                "",
                "Hướng dẫn sử dụng:",
                "- 'standard': Phù hợp cho tìm kiếm từ khóa và vector search",
                "- 'hyde': Phù hợp cho tìm kiếm ngữ nghĩa phức tạp"
            ])
            
            return "\n".join(result_lines)
        
        except Exception as e:
            error_msg = f"Lỗi khi chuyển đổi câu hỏi: {str(e)}"
            logger.error(error_msg)
            return error_msg


    def grade_retrieval(self, query: str, retriever_type: str = "vector") -> str:
        """
        Process a query and return grading results for retrieved documents.
        
        Args:
            query: Question to grade documents for
            retriever_type: Type of retriever to use (default is "vector")
            
        Returns:
            String showing retrieved documents and their relevance scores
        """
        if not query:
            return "Vui lòng nhập câu hỏi."
        
        try:
            # Get retriever and retrieve nodes for the question
            retriever = self.retrievers.get(retriever_type)
            if not retriever:
                return f"Retriever type '{retriever_type}' not found."
                
            retrieved_nodes = retriever.retrieve(query)
            
            # Use the first available generator to grade documents
            if not self.generators:
                return "No generators available."
                
            generator = self.generators[-1]
            
            # Grade the documents
            logger.info("Grading document relevance...")
            scores = generator.grade_documents_relevance(query, retrieved_nodes)
            
            # Format the results
            result_lines = [
                f"Câu hỏi: {query}",
                f"Retriever sử dụng: {retriever_type}",
                f"Số lượng tài liệu được truy xuất: {len(retrieved_nodes)}",
                "",
                "Kết quả đánh giá độ liên quan của từng tài liệu:",
                "=" * 60
            ]
            
            for i, (node, score) in enumerate(zip(retrieved_nodes, scores), 1):
                metadata = node.node.metadata
                doc_id = metadata.get("document_id", "Unknown")
                article = metadata.get("article", "")
                chapter = metadata.get("chapter", "")
                
                # Get document snippet
                doc_text = node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text
                
                # Format relevance status
                relevance_status = "✅ LIÊN QUAN" if score.lower() == "yes" else "❌ KHÔNG LIÊN QUAN"
                
                result_lines.extend([
                    f"\nTài liệu {i}:",
                    f"Độ liên quan: {relevance_status} ({score})",
                    f"Văn bản: {doc_id}",
                ])
                
                if chapter:
                    result_lines.append(f"Chương: {chapter}")
                if article:
                    result_lines.append(f"Điều: {article}")
                    
                result_lines.extend([
                    f"Trích đoạn: {doc_text}",
                    "-" * 40
                ])
            
            # Add summary
            relevant_count = sum(1 for score in scores if score.lower() == "yes")
            total_count = len(scores)
            
            result_lines.extend([
                "",
                "TÓM TẮT:",
                f"Tổng số tài liệu: {total_count}",
                f"Tài liệu liên quan: {relevant_count}",
                f"Tài liệu không liên quan: {total_count - relevant_count}",
                f"Tỷ lệ liên quan: {relevant_count/total_count*100:.1f}%" if total_count > 0 else "0%"
            ])
            
            return "\n".join(result_lines)
        
        except Exception as e:
            error_msg = f"Lỗi khi đánh giá tài liệu: {str(e)}"
            logger.error(error_msg)
            return error_msg


def main():
    """Main function to run the evaluation."""
    try:
        evaluator = RAGEvaluator()
        evaluator.initialize_generators()
        evaluator.initialize_embed_model()
        # evaluator.load_qa_data()
        evaluator.load_indexes(load_from='local')
        evaluator.setup_retrievers()

        # You can run these phases independently:
        
        # Phase 1: Retrieval only
        # evaluator.run_retrieval_phase()
        
        # Phase 2: Generation (can be run later or on a different machine)
        # Comment out if you only want to do retrieval now
        # evaluator.run_generation_phase(retriever_names=["vector"])
        
        # Phase 3: Evaluation (can be run after generation is complete)
        # Comment out if you only want to do retrieval and/or generation
        # evaluator.run_evaluation_phase()
        
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()