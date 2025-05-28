"""
Flask API for Vietnamese legal document QA system using RAG.
"""

import os
import sys
import logging
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from typing import Dict, Any
import traceback

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import the evaluator
from eval import RAGEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chat_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global evaluator instance
evaluator = None

def initialize_rag_system():
    """Initialize the RAG system once at startup."""
    global evaluator
    try:
        logger.info("Initializing RAG system...")
        evaluator = RAGEvaluator()
        
        # Initialize all components
        evaluator.initialize_generators()
        evaluator.initialize_embed_model()
        evaluator.load_indexes(load_from='local')
        evaluator.setup_retrievers()
        
        logger.info("RAG system initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/', methods=['GET'])
def home():
    """Home page with simple chat interface."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vietnamese Legal QA System</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 900px; margin: 0 auto; }
            .chat-box { border: 1px solid #ccc; height: 400px; overflow-y: scroll; padding: 10px; margin: 20px 0; white-space: pre-wrap; }
            .metadata-box { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background: #f8f9fa; }
            .input-area { display: flex; gap: 10px; margin-bottom: 10px; }
            .controls { display: flex; gap: 15px; margin-bottom: 20px; align-items: center; flex-wrap: wrap; }
            .pipeline-controls { margin-bottom: 15px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; }
            #query { flex: 1; padding: 10px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 4px; }
            button:hover { background: #0056b3; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #e3f2fd; text-align: right; }
            .bot { background: #f5f5f5; }
            .grading { background: #fff3e0; }
            .routing { background: #e8f5e8; }
            .rephrasing { background: #f3e5f5; }
            .error { background: #ffebee; color: #c62828; }
            .loading { color: #666; font-style: italic; }
            select { padding: 8px; border-radius: 4px; border: 1px solid #ccc; }
            label { font-weight: bold; }
            .mode-selection { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
            .arch-selection { display: flex; gap: 10px; align-items: center; }
            .pipeline-option { margin: 5px 0; }
            input[type="radio"] { margin-right: 5px; }
            input[type="checkbox"] { margin-right: 5px; }
            .conditional-control { margin-top: 10px; }
            .pipeline-title { font-weight: bold; margin-bottom: 10px; color: #333; }
            .metadata-title { font-weight: bold; color: #495057; margin-bottom: 10px; }
            .metadata-item { margin: 5px 0; padding: 5px; border-left: 3px solid #6c757d; padding-left: 10px; }
            .metadata-value { font-family: monospace; background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
            .metadata-section { margin: 10px 0; }
            .metadata-nodes { max-height: 200px; overflow-y: auto; font-size: 0.9em; }
            .clear-metadata { float: right; padding: 5px 10px; font-size: 0.8em; background: #6c757d; }
            .clear-metadata:hover { background: #5a6268; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vietnamese Legal Document QA System</h1>
            <p>Ask questions about Vietnamese legal documents, regulations, and circulars.</p>
            
            <div class="controls">
                <div>
                    <label for="retriever">Retriever Type:</label>
                    <select id="retriever">
                        <option value="vector">Vector Search</option>
                        <option value="bm25">BM25 (Keyword)</option>
                        <option value="hybrid_fusion">Hybrid Fusion</option>
                    </select>
                </div>
                
                <div class="mode-selection">
                    <label>Mode:</label>
                    <label><input type="radio" name="mode" value="answer" checked> Answer</label>
                    <label><input type="radio" name="mode" value="grade"> Grade Documents</label>
                    <label><input type="radio" name="mode" value="route"> Route Question</label>
                    <label><input type="radio" name="mode" value="rephrase"> Rephrase Query</label>
                </div>
            </div>
            
            <!-- Pipeline Controls (only visible in answer mode) -->
            <div class="pipeline-controls" id="pipelineControls">
                <div class="pipeline-title">🔧 RAG Pipeline Options:</div>
                
                <div class="pipeline-option">
                    <label><input type="checkbox" id="useRouting"> Use Query Routing</label>
                    <small>(Check if the question needs legal information retrieval)</small>
                </div>
                
                <div class="pipeline-option">
                    <label><input type="checkbox" id="useRephrasing"> Use Query Rephrasing</label>
                    <div id="rephrasingOptions" style="margin-left: 20px; display: none;">
                        <label><input type="radio" name="rephrase_arch" value="standard" checked> Standard</label>
                        <label><input type="radio" name="rephrase_arch" value="hyde"> HyDE</label>
                    </div>
                </div>
                
                <div class="pipeline-option">
                    <label><input type="checkbox" id="useGrading"> Grade Retrieved Documents</label>
                    <small>(Filter out irrelevant documents)</small>
                </div>
                
                <div class="pipeline-option">
                    <label><input type="checkbox" id="gradeAnswer"> Grade Answer Usefulness</label>
                    <small>(Evaluate how well the answer addresses the question)</small>
                </div>
            </div>
            
            <!-- Conditional control for rephrase mode -->
            <div class="conditional-control" id="rephraseControl" style="display: none;">
                <div class="arch-selection">
                    <label>Rephrasing Architecture:</label>
                    <label><input type="radio" name="arch" value="standard" checked> Standard</label>
                    <label><input type="radio" name="arch" value="hyde"> HyDE</label>
                </div>
                <p><small>
                    <strong>Standard:</strong> Rephrase using synonyms and related terms<br>
                    <strong>HyDE:</strong> Generate hypothetical answer paragraph
                </small></p>
            </div>
            
            <div class="chat-box" id="chatBox"></div>
            
            <!-- Metadata Display -->
            <div class="metadata-box" id="metadataBox" style="display: none;">
                <div class="metadata-title">
                    📊 Answer Generation Metadata
                    <button class="clear-metadata" onclick="clearMetadata()">Clear</button>
                </div>
                <div id="metadataContent"></div>
            </div>
            
            <div class="input-area">
                <input type="text" id="query" placeholder="Enter your question about Vietnamese legal documents..." 
                       onkeypress="if(event.key==='Enter') processQuery()">
                <button onclick="processQuery()" id="submitBtn">Ask Question</button>
            </div>
        </div>

        <script>
            function addMessage(content, type) {
                const chatBox = document.getElementById('chatBox');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}`;
                messageDiv.innerHTML = content.replace(/\\n/g, '<br>');
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function displayMetadata(metadata, pipelineOptions) {
                if (!metadata || Object.keys(metadata).length === 0) {
                    return;
                }

                const metadataBox = document.getElementById('metadataBox');
                const metadataContent = document.getElementById('metadataContent');
                
                let html = '';
                
                // Pipeline Options Used
                if (pipelineOptions) {
                    html += '<div class="metadata-section">';
                    html += '<strong>🔧 Pipeline Configuration:</strong><br>';
                    html += `<div class="metadata-item">Routing: <span class="metadata-value">${pipelineOptions.routing ? 'Enabled' : 'Disabled'}</span></div>`;
                    html += `<div class="metadata-item">Rephrasing: <span class="metadata-value">${pipelineOptions.rephrasing ? pipelineOptions.rephrase_arch.toUpperCase() : 'Disabled'}</span></div>`;
                    html += `<div class="metadata-item">Document Grading: <span class="metadata-value">${pipelineOptions.grading ? 'Enabled' : 'Disabled'}</span></div>`;
                    html += `<div class="metadata-item">Answer Grading: <span class="metadata-value">${pipelineOptions.answer_grading ? 'Enabled' : 'Disabled'}</span></div>`;
                    html += '</div>';
                }

                // Routing Results
                if (metadata.retrieve) {
                    html += '<div class="metadata-section">';
                    html += '<strong>🔄 Query Routing:</strong><br>';
                    html += `<div class="metadata-item">Decision: <span class="metadata-value">${metadata.retrieve.toUpperCase()}</span></div>`;
                    html += '</div>';
                }

                // Rephrasing Results
                if (metadata.rephrase) {
                    html += '<div class="metadata-section">';
                    html += '<strong>✏️ Query Rephrasing:</strong><br>';
                    html += `<div class="metadata-item">Architecture: <span class="metadata-value">${metadata.rephrase_arch ? metadata.rephrase_arch.toUpperCase() : 'N/A'}</span></div>`;
                    html += `<div class="metadata-item">Rephrased Query: <span class="metadata-value">${metadata.rephrase}</span></div>`;
                    html += '</div>';
                }

                // Retrieval Results
                if (metadata.retriever_type) {
                    html += '<div class="metadata-section">';
                    html += '<strong>🔍 Document Retrieval:</strong><br>';
                    html += `<div class="metadata-item">Retriever: <span class="metadata-value">${metadata.retriever_type}</span></div>`;
                    if (metadata.retrieved_nodes) {
                        html += `<div class="metadata-item">Documents Retrieved: <span class="metadata-value">${metadata.retrieved_nodes.length}</span></div>`;
                    }
                    html += '</div>';
                }

                // Document Grading Results
                if (metadata.relevance) {
                    html += '<div class="metadata-section">';
                    html += '<strong>📊 Document Relevance Grading:</strong><br>';
                    const totalCount = metadata.relevance.length;
                    const relevantCount = metadata.relevance.filter(score => score.toLowerCase() === 'yes').length;
                    html += `<div class="metadata-item">Relevant Documents: <span class="metadata-value">${relevantCount}/${totalCount}</span></div>`;
                    html += `<div class="metadata-item">Relevance Rate: <span class="metadata-value">${((relevantCount/totalCount)*100).toFixed(1)}%</span></div>`;
                    html += '</div>';
                }

                // Answer Usefulness Results
                if (metadata.usefulness_score) {
                    html += '<div class="metadata-section">';
                    html += '<strong>📈 Answer Usefulness:</strong><br>';
                    html += `<div class="metadata-item">Score: <span class="metadata-value">${metadata.usefulness_score}/5</span></div>`;
                    if (metadata.usefulness_explanation) {
                        html += `<div class="metadata-item">Explanation: <span class="metadata-value">${metadata.usefulness_explanation}</span></div>`;
                    }
                    html += '</div>';
                }

                // Retrieved Nodes (if available and not too many)
                if (metadata.retrieved_nodes && metadata.retrieved_nodes.length > 0) {
                    html += '<div class="metadata-section">';
                    html += '<strong>📄 Retrieved Document Snippets:</strong><br>';
                    html += '<div class="metadata-nodes">';
                    metadata.retrieved_nodes.slice(0, 3).forEach((node, index) => {
                        const snippet = node.length > 150 ? node.substring(0, 150) + '...' : node;
                        html += `<div class="metadata-item">Document ${index + 1}: <span class="metadata-value">${snippet}</span></div>`;
                    });
                    if (metadata.retrieved_nodes.length > 3) {
                        html += `<div class="metadata-item"><em>... and ${metadata.retrieved_nodes.length - 3} more documents</em></div>`;
                    }
                    html += '</div>';
                    html += '</div>';
                }

                metadataContent.innerHTML = html;
                metadataBox.style.display = 'block';
            }

            function clearMetadata() {
                const metadataBox = document.getElementById('metadataBox');
                metadataBox.style.display = 'none';
            }

            function getSelectedMode() {
                return document.querySelector('input[name="mode"]:checked').value;
            }

            function getSelectedArch() {
                const archRadio = document.querySelector('input[name="arch"]:checked');
                return archRadio ? archRadio.value : 'standard';
            }

            function getSelectedRephraseArch() {
                const archRadio = document.querySelector('input[name="rephrase_arch"]:checked');
                return archRadio ? archRadio.value : 'standard';
            }

            function updateButtonText() {
                const mode = getSelectedMode();
                const button = document.getElementById('submitBtn');
                const queryInput = document.getElementById('query');
                const retrieverDiv = document.getElementById('retriever').parentElement;
                const rephraseControl = document.getElementById('rephraseControl');
                const pipelineControls = document.getElementById('pipelineControls');
                
                // Hide/show controls based on mode
                if (mode === 'answer') {
                    button.textContent = 'Ask Question';
                    queryInput.placeholder = 'Enter your question about Vietnamese legal documents...';
                    retrieverDiv.style.display = 'block';
                    rephraseControl.style.display = 'none';
                    pipelineControls.style.display = 'block';
                } else if (mode === 'grade') {
                    button.textContent = 'Grade Documents';
                    queryInput.placeholder = 'Enter your question to grade document relevance...';
                    retrieverDiv.style.display = 'block';
                    rephraseControl.style.display = 'none';
                    pipelineControls.style.display = 'none';
                } else if (mode === 'route') {
                    button.textContent = 'Route Question';
                    queryInput.placeholder = 'Enter any question to test routing decision...';
                    retrieverDiv.style.display = 'none';
                    rephraseControl.style.display = 'none';
                    pipelineControls.style.display = 'none';
                } else if (mode === 'rephrase') {
                    button.textContent = 'Rephrase Query';
                    queryInput.placeholder = 'Enter your question to test query rephrasing...';
                    retrieverDiv.style.display = 'none';
                    rephraseControl.style.display = 'block';
                    pipelineControls.style.display = 'none';
                }

                // Clear metadata when switching modes (except for answer mode)
                if (mode !== 'answer') {
                    clearMetadata();
                }
            }

            // Show/hide rephrasing options based on checkbox
            document.getElementById('useRephrasing').addEventListener('change', function() {
                const rephrasingOptions = document.getElementById('rephrasingOptions');
                rephrasingOptions.style.display = this.checked ? 'block' : 'none';
            });

            // Update button text when mode changes
            document.querySelectorAll('input[name="mode"]').forEach(radio => {
                radio.addEventListener('change', updateButtonText);
            });

            async function processQuery() {
                const queryInput = document.getElementById('query');
                const retrieverSelect = document.getElementById('retriever');
                const query = queryInput.value.trim();
                const mode = getSelectedMode();
                
                if (!query) {
                    alert('Please enter a question');
                    return;
                }

                // Add user message
                addMessage(query, 'user');
                queryInput.value = '';

                // Clear previous metadata for answer mode
                if (mode === 'answer') {
                    clearMetadata();
                }

                // Add loading message
                let loadingMessage;
                switch(mode) {
                    case 'grade':
                        loadingMessage = 'Grading documents...';
                        break;
                    case 'route':
                        loadingMessage = 'Routing question...';
                        break;
                    case 'rephrase':
                        loadingMessage = 'Rephrasing query...';
                        break;
                    default:
                        loadingMessage = 'Processing with RAG pipeline...';
                }
                addMessage(loadingMessage, 'loading');

                try {
                    let endpoint;
                    let requestBody = { query: query };
                    
                    switch(mode) {
                        case 'grade':
                            endpoint = '/grade';
                            requestBody.retriever_type = retrieverSelect.value;
                            break;
                        case 'route':
                            endpoint = '/route';
                            break;
                        case 'rephrase':
                            endpoint = '/rephrase';
                            requestBody.arch = getSelectedArch();
                            break;
                        default:
                            endpoint = '/chat';
                            requestBody.retriever_type = retrieverSelect.value;
                            
                            // Add pipeline options for answer mode
                            requestBody.use_routing = document.getElementById('useRouting').checked;
                            requestBody.use_rephrasing = document.getElementById('useRephrasing').checked;
                            requestBody.rephrase_arch = getSelectedRephraseArch();
                            requestBody.use_grading = document.getElementById('useGrading').checked;
                            requestBody.grade_answer = document.getElementById('gradeAnswer').checked;
                    }

                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestBody)
                    });

                    const data = await response.json();

                    // Remove loading message
                    const messages = document.querySelectorAll('.message.loading');
                    if (messages.length > 0) {
                        messages[messages.length - 1].remove();
                    }

                    if (data.success) {
                        let messageType;
                        switch(mode) {
                            case 'grade':
                                messageType = 'grading';
                                break;
                            case 'route':
                                messageType = 'routing';
                                break;
                            case 'rephrase':
                                messageType = 'rephrasing';
                                break;
                            default:
                                messageType = 'bot';
                        }
                        addMessage(data.response, messageType);

                        // Display metadata for answer mode
                        if (mode === 'answer' && data.answer_metadata) {
                            displayMetadata(data.answer_metadata, data.pipeline_options);
                        }
                    } else {
                        addMessage(`Error: ${data.error}`, 'error');
                    }
                } catch (error) {
                    // Remove loading message
                    const messages = document.querySelectorAll('.message.loading');
                    if (messages.length > 0) {
                        messages[messages.length - 1].remove();
                    }
                    addMessage(`Network error: ${error.message}`, 'error');
                }
            }

            // Initialize button text
            updateButtonText();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint for answering questions with full RAG pipeline."""
    global evaluator
    
    # Check if system is initialized
    if evaluator is None:
        return jsonify({
            'success': False,
            'error': 'RAG system not initialized'
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        query = data.get('query', '').strip()
        retriever_type = data.get('retriever_type', 'vector')
        cutoff = data.get('cutoff', 0.0)
        
        # Pipeline options
        use_routing = data.get('use_routing', False)
        use_rephrasing = data.get('use_rephrasing', False)
        rephrase_arch = data.get('rephrase_arch', 'standard')
        use_grading = data.get('use_grading', False)
        grade_answer = data.get('grade_answer', False)
        
        # Validate input
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        if retriever_type not in evaluator.retrievers:
            return jsonify({
                'success': False,
                'error': f'Invalid retriever type. Available: {list(evaluator.retrievers.keys())}'
            }), 400
        
        if rephrase_arch not in ['standard', 'hyde']:
            return jsonify({
                'success': False,
                'error': 'Invalid rephrasing architecture. Must be "standard" or "hyde"'
            }), 400
        
        logger.info(f"Processing query with full pipeline: routing={use_routing}, rephrasing={use_rephrasing}, grading={use_grading}, answer_grading={grade_answer}")
        
        # Process the query with full pipeline
        result, answer_metadata = evaluator.answer_query(
            query=query,
            retriever_type=retriever_type,
            cutoff=cutoff,
            use_routing=use_routing,
            use_rephrasing=use_rephrasing,
            rephrase_arch=rephrase_arch,
            use_grading=use_grading,
            grade_answer=grade_answer
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'query': query,
            'response': result,
            'answer_metadata': answer_metadata,
            'pipeline_options': {
                'routing': use_routing,
                'rephrasing': use_rephrasing,
                'rephrase_arch': rephrase_arch,
                'grading': use_grading,
                'answer_grading': grade_answer
            }
        }

        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/route', methods=['POST'])
def route():
    """Route question endpoint."""
    global evaluator
    
    # Check if system is initialized
    if evaluator is None:
        return jsonify({
            'success': False,
            'error': 'RAG system not initialized'
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        query = data.get('query', '').strip()
        
        # Validate input
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        logger.info(f"Routing question: {query[:100]}...")
        
        # Route the question
        result = evaluator.route_query(query=query)
        
        # Prepare response
        response_data = {
            'success': True,
            'response': result,
            'query': query,
            'mode': 'routing'
        }
        
        logger.info(f"Successfully routed question")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error routing question: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/rephrase', methods=['POST'])
def rephrase():
    """Rephrase query endpoint."""
    global evaluator
    
    # Check if system is initialized
    if evaluator is None:
        return jsonify({
            'success': False,
            'error': 'RAG system not initialized'
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        query = data.get('query', '').strip()
        arch = data.get('arch', 'standard')
        
        # Validate input
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        if arch not in ['standard', 'hyde']:
            return jsonify({
                'success': False,
                'error': 'Invalid architecture. Must be "standard" or "hyde"'
            }), 400
        
        logger.info(f"Rephrasing query with {arch} architecture: {query[:100]}...")
        
        # Rephrase the query
        result = evaluator.rephrase_query(query=query, arch=arch)
        
        # Prepare response
        response_data = {
            'success': True,
            'response': result,
            'query': query,
            'architecture': arch,
            'mode': 'rephrasing'
        }
        
        logger.info(f"Successfully rephrased query using {arch} architecture")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error rephrasing query: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/grade', methods=['POST'])
def grade():
    """Grade document relevance endpoint."""
    global evaluator
    
    # Check if system is initialized
    if evaluator is None:
        return jsonify({
            'success': False,
            'error': 'RAG system not initialized'
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        query = data.get('query', '').strip()
        retriever_type = data.get('retriever_type', 'vector')
        
        # Validate input
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        if retriever_type not in evaluator.retrievers:
            return jsonify({
                'success': False,
                'error': f'Invalid retriever type. Available: {list(evaluator.retrievers.keys())}'
            }), 400
        
        logger.info(f"Grading documents with {retriever_type} retriever for query: {query[:100]}...")
        
        # Grade the documents
        result = evaluator.grade_retrieval(
            query=query,
            retriever_type=retriever_type
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'response': result,
            'retriever_used': retriever_type,
            'query': query,
            'mode': 'grading'
        }
        
        logger.info(f"Successfully graded documents for query")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error grading documents: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global evaluator
    if evaluator is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'RAG system not initialized'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'message': 'RAG system is running',
        'retrievers': list(evaluator.retrievers.keys()) if evaluator.retrievers else [],
        'generators': len(evaluator.generators) if evaluator.generators else 0
    })


@app.route('/retrievers', methods=['GET'])
def get_retrievers():
    """Get available retriever types."""
    global evaluator
    
    if evaluator is None:
        return jsonify({
            'success': False,
            'error': 'RAG system not initialized'
        }), 503
    
    return jsonify({
        'success': True,
        'retrievers': list(evaluator.retrievers.keys()),
        'default': 'vector'
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    global evaluator
    
    if evaluator is None:
        return jsonify({
            'success': False,
            'error': 'RAG system not initialized'
        }), 503
    
    stats = {
        'success': True,
        'parent_nodes': len(evaluator.parent_nodes) if evaluator.parent_nodes else 0,
        'child_nodes': len(evaluator.child_nodes) if evaluator.child_nodes else 0,
        'retrievers': list(evaluator.retrievers.keys()),
        'generators': len(evaluator.generators),
        'embed_model': evaluator.embed_model.__class__.__name__ if evaluator.embed_model else None
    }
    
    return jsonify(stats)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


def main():
    """Main function to run the Flask app."""
    # Initialize the RAG system
    if not initialize_rag_system():
        logger.error("Failed to initialize RAG system. Exiting.")
        sys.exit(1)
    
    # Run the Flask app
    logger.info("Starting Flask API server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to True for development
        threaded=True
    )


if __name__ == "__main__":
    main()