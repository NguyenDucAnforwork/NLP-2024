"""
Functions for parsing Vietnamese legal documents into structured format.
"""
import re
from typing import Dict, Any, List, Tuple

def extract_chapters(text: str) -> Dict[str, str]:
    """
    Extract chapters from legal text.
    
    Args:
        text: Legal text
        
    Returns:
        Dictionary mapping chapter names to their content
    """
    # Pattern for chapter headers like "Chương I:" or "Chương 1:"
    chapter_pattern = r'(Chương\s+[IVXLCDMivxlcdm0-9]+\s*:?[^Đ]*?)(?=Chương\s+[IVXLCDMivxlcdm0-9]+\s*:|$)'
    
    chapters = {}
    matches = re.finditer(chapter_pattern, text, re.DOTALL)
    
    for match in matches:
        chapter_text = match.group(1).strip()
        # Extract chapter name (e.g., "Chương I", "Chương 1")
        chapter_name_match = re.match(r'(Chương\s+[IVXLCDMivxlcdm0-9]+)', chapter_text)
        if chapter_name_match:
            chapter_name = chapter_name_match.group(1)
            chapters[chapter_name] = chapter_text
    
    return chapters

def extract_chapter_title(chapter_text: str) -> str:
    """
    Extract the title of a chapter from its content.
    
    Args:
        chapter_text: Text content of the chapter
        
    Returns:
        Chapter title
    """
    # Extract title after the chapter number
    title_match = re.search(r'Chương\s+[IVXLCDMivxlcdm0-9]+\s*:?\s*(.*?)(?=\n\n|\n\s*Điều|\Z)', chapter_text, re.DOTALL)
    
    if title_match:
        return title_match.group(1).strip()
    
    return ""

def extract_articles(text: str) -> Dict[str, Any]:
    """
    Extract articles from legal text.
    
    Args:
        text: Legal text
        
    Returns:
        Dictionary mapping article names to their content
    """
    # Pattern for article headers like "Điều 1:" or "Điều 1."
    article_pattern = r'(Điều\s+\d+\s*[:.]\s*[^Đ]*?)(?=Điều\s+\d+|$)'
    
    articles = {}
    matches = re.finditer(article_pattern, text, re.DOTALL)
    
    for match in matches:
        article_text = match.group(1).strip()
        # Extract article name (e.g., "Điều 1")
        article_name_match = re.match(r'(Điều\s+\d+)', article_text)
        if article_name_match:
            article_name = article_name_match.group(1)
            # Extract article title and content
            title_content_match = re.search(r'Điều\s+\d+\s*[:.]\s*(.*?)(?=\n\n|\n|$)(.*)', article_text, re.DOTALL)
            
            if title_content_match:
                title = title_content_match.group(1).strip()
                content = title_content_match.group(2).strip()
                
                articles[article_name] = {
                    "title": title,
                    "content": f"{title}\n\n{content}"
                }
            else:
                articles[article_name] = {
                    "content": article_text
                }
    
    return articles

def merge_short_articles(articles: Dict[str, Any], min_length: int, max_length: int) -> Dict[str, Any]:
    """
    Merge short articles to reach a minimum length.
    
    Args:
        articles: Dictionary of articles
        min_length: Minimum article length
        max_length: Maximum article length
        
    Returns:
        Dictionary with merged articles
    """
    if min_length <= 0:
        return articles
    
    # Sort article names
    article_names = sorted(articles.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    
    merged_articles = {}
    current_group = []
    current_length = 0
    parent_links = []
    
    for article_name in article_names:
        article = articles[article_name]
        article_content = article.get('content', '')
        content_length = len(article_content)
        
        # If adding this article would exceed max_length, finalize the current group
        if current_group and (current_length + content_length > max_length):
            merged_name = "+".join(current_group)
            merged_content = "\n\n".join([articles[name].get('content', '') for name in current_group])
            
            merged_articles[merged_name] = {
                "content": merged_content,
                "parent_links": parent_links
            }
            
            # Reset for the next group
            current_group = []
            current_length = 0
            parent_links = []
        
        # Add article to current group
        current_group.append(article_name)
        current_length += content_length
        parent_links.append(article_name)
        
        # If the group is long enough and there's only one article, just keep it as is
        if current_length >= min_length and len(current_group) == 1:
            merged_articles[article_name] = article
            current_group = []
            current_length = 0
            parent_links = []
    
    # Handle any remaining articles in the current group
    if current_group:
        if len(current_group) == 1:
            # Just one article, keep it as is
            article_name = current_group[0]
            merged_articles[article_name] = articles[article_name]
        else:
            # Merge remaining articles
            merged_name = "+".join(current_group)
            merged_content = "\n\n".join([articles[name].get('content', '') for name in current_group])
            
            merged_articles[merged_name] = {
                "content": merged_content,
                "parent_links": parent_links
            }
    
    return merged_articles

def parse_text_to_structure(text: str, min_length: int = 400, max_length: int = 2000) -> Dict[str, Any]:
    """
    Parse Vietnamese legal text into a structured format.
    
    Args:
        text: Raw legal text
        min_length: Minimum length for merging segments
        max_length: Maximum length for merging segments
        
    Returns:
        Dictionary with structured legal text
    """
    # Extract chapters and articles
    chapters = extract_chapters(text)
    
    # If no chapters found, try to extract articles directly
    if not chapters:
        articles = extract_articles(text)
        
        # Merge short articles if needed
        if min_length > 0:
            articles = merge_short_articles(articles, min_length, max_length)
        
        return articles
    
    # Process chapters and their articles
    structured_data = {}
    
    for chapter_name, chapter_content in chapters.items():
        # Extract chapter title
        chapter_title = extract_chapter_title(chapter_content)
        
        # Extract articles in this chapter
        chapter_articles = extract_articles(chapter_content)
        
        # Merge short articles if needed
        if min_length > 0:
            chapter_articles = merge_short_articles(chapter_articles, min_length, max_length)
        
        # Add chapter to structured data
        structured_data[chapter_name] = {"title": chapter_title}
        structured_data[chapter_name].update(chapter_articles)
    
    return structured_data