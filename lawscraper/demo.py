"""
Gradio demo for Vietnamese legal document question answering with parent-child chunking
"""
import os
import sys
import logging
import gradio as gr
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import our custom modules
from src.backend.vector_indexes import ParentVectorIndex, ChildVectorIndex
from llama_index.core.retrievers import AutoMergingRetriever # no need to import BaseRetriever
from src.backend.generators import LLMGenerator

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode, BaseNode

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if Gemini API key is set
if not os.getenv("GEMINI_API_KEY"):
    logger.warning("GEMINI_API_KEY not found in environment variables.")
    logger.warning("Please set it in the .env file before running this demo.")

# Check if Qdrant credentials are set
if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
    logger.warning("QDRANT_URL or QDRANT_API_KEY not found in environment variables.")
    logger.warning("Please set them in the .env file before running this demo.")

# Global variables to store our index
retriever = None
parent_index = None
child_index = None
generator = None

def initialize_index():
    """Initialize or load the index from Qdrant Cloud"""
    global retriever, parent_index, child_index, generator
    
    try:
        logger.info("Loading index from Qdrant Cloud...")
        
        # Initialize parent index
        parent_idx = ParentVectorIndex(
            collection_name="vietnamese_legal_parent",
            embed_model_path="./data/embeddings/bge-m3"
        )
        parent_index, parent_nodes = parent_idx.load_index_from_cloud()
        
        # Initialize child index
        child_idx = ChildVectorIndex(
            collection_name="vietnamese_legal_child",
            embed_model_path="./data/embeddings/bge-m3"
        )
        child_index, child_nodes = child_idx.load_index_from_cloud()
        
        # Create storage context and add all nodes        
        docstore = SimpleDocumentStore()
        docstore.add_documents(parent_nodes)
        
        # Add child nodes and try to establish parent-child relationships using metadata
        for child in child_nodes:
            if 'parent_id' in child.metadata:
                parent_id = child.metadata['parent_id']
                try:
                    parent_node = docstore.get_document(parent_id)
                    if isinstance(parent_node, BaseNode):
                        child.parent_node = {"node_id": parent_id}
                except:
                    logger.warning(f"Could not find parent node with ID {parent_id}")
        
        docstore.add_documents(child_nodes)
        
        # Create storage context with the docstore
        storage_context = StorageContext.from_defaults(docstore=docstore)
        
        # Create the base retriever for the child index
        base_retriever = child_index.as_retriever(similarity_top_k=6)
        
        # Create auto-merging retriever
        retriever = AutoMergingRetriever(
            vector_retriever=base_retriever,
            storage_context=storage_context,
            verbose=True
        )
        
        # Initialize the generator
        generator = LLMGenerator(
            model_name="models/gemini-2.0-flash",
            temperature=0.7
        )
        
        logger.info("Successfully connected to Qdrant Cloud index!")
        return "Successfully connected to Qdrant Cloud index!"
    except Exception as e:
        error_msg = f"Error connecting to Qdrant Cloud: {str(e)}"
        logger.error(error_msg)
        return error_msg

def answer_query(query):
    """Process a user query and return the answer with sources"""
    global retriever, generator
    
    # Check if index is initialized
    if retriever is None or generator is None:
        return "Please initialize the index first by clicking the 'Connect to Cloud' button."
    
    if not query.strip():
        return "Please enter a question."
    
    try:
        # Retrieve nodes for the query
        retrieved_nodes = retriever.retrieve(query)
        
        # Generate response using LLM
        response = generator.generate_response(query, retrieved_nodes)
        
        # Format source information
        source_info = []
        for i, source in enumerate(retrieved_nodes[:3], 1):  # Show top 3 sources
            metadata = source.node.metadata
            doc_id = metadata.get("document_id", "Unknown")
            article = metadata.get("article", "")
            chapter = metadata.get("chapter", "")
            
            source_text = source.node.text[:150] + "..." if len(source.node.text) > 150 else source.node.text
            
            source_str = f"**Source {i}:** Document: {doc_id}\n"
            if chapter:
                source_str += f"Chapter: {chapter}\n"
            if article:
                source_str += f"Article: {article}\n"
            source_str += f"Text preview: {source_text}\n\n"
            source_info.append(source_str)
        
        # Create final output
        sources_text = "\n".join(source_info)
        result = f"### Answer:\n{response}\n\n### Sources:\n{sources_text}"
        
        return result
    
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Setup Gradio interface
with gr.Blocks(title="Vietnamese Legal Document Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Vietnamese Legal Document Assistant")
    gr.Markdown("This demo uses a parent-child chunking strategy with hybrid search to answer questions about Vietnamese legal documents.")
    
    with gr.Row():
        with gr.Column(scale=2):
            init_button = gr.Button("Connect to Cloud", variant="primary")
            status_text = gr.Textbox(label="Connection Status", interactive=False)
            
            # Setup for interaction
            init_button.click(initialize_index, inputs=[], outputs=[status_text])
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### Cloud Features:
            - Pre-indexed documents in Qdrant Cloud
            - Fast retrieval with hybrid search
            - No local indexing required
            """)
    
    gr.Markdown("---")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Ask a question about Vietnamese legal documents",
            placeholder="Thông tư này có hiệu lực từ ngày nào?",
            lines=2
        )
        query_button = gr.Button("Submit Question", variant="primary")
    
    answer_output = gr.Markdown(label="Answer")
    
    # Set up the query button
    query_button.click(answer_query, inputs=[query_input], outputs=[answer_output])
    query_input.submit(answer_query, inputs=[query_input], outputs=[answer_output])

if __name__ == "__main__":
    logger.info("Starting Vietnamese Legal Document Assistant...")
    logger.info("Connecting to cloud index (this may take a moment)...")
    initialize_message = initialize_index()
    logger.info(initialize_message)
    logger.info("Launching Gradio interface...")
    demo.launch(share=True)