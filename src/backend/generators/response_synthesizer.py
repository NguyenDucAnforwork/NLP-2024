"""
Response synthesizer for generating coherent responses.
"""
from typing import Any, Optional

from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

def create_response_synthesizer(
    llm: Any, 
    response_mode: str = "compact",
    text_qa_template: Optional[PromptTemplate] = None
) -> Any:
    """
    Create a response synthesizer with the given parameters.
    
    Args:
        llm: Language model to use
        response_mode: Mode for response synthesis
        text_qa_template: Template for text QA
        
    Returns:
        Response synthesizer
    """
    if text_qa_template is None:
        # Cải thiện prompt template để tránh "I don't know"
        text_qa_template = PromptTemplate(
            """Bạn là một trợ lý trả lời câu hỏi về các văn bản pháp luật, thông tư, quy định của Việt Nam. 
            Sử dụng các đoạn văn bản tham khảo dưới đây để trả lời câu hỏi một cách đầy đủ và chính xác.
            
            Nếu bạn tìm thấy thông tin liên quan trong các đoạn văn bản tham khảo, hãy sử dụng chúng để trả lời chi tiết.
            Tránh trả lời "Tôi không biết" hoặc "Tôi không có thông tin" khi đã có thông tin liên quan trong các đoạn văn bản.
            Nếu câu hỏi không liên quan đến nội dung trong các đoạn văn bản, hãy trả lời rằng bạn không tìm thấy thông tin liên quan để trả lời.
            
            Trả lời bằng tiếng Việt, sử dụng giọng điệu chuyên nghiệp, trích dẫn các điều khoản cụ thể khi cần thiết, và trình bày mạch lạc, dễ hiểu.
            
            Đoạn văn bản tham khảo:
            {context_str}
            
            Câu hỏi: {query_str}
            
            Trả lời (dựa trên văn bản tham khảo):
            """
        )
    
    return get_response_synthesizer(
        response_mode=response_mode,
        llm=llm,
        text_qa_template=text_qa_template
    )