"""
Auto-merging retriever implementation for improved context retrieval.
"""
from typing import List, Dict, Tuple, Any, Optional, cast
from collections import defaultdict

from llama_index.core import QueryBundle, StorageContext
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.core.utils import truncate_text
# from llama_index.core.indices.query.schema import MetadataMode

from .base_retriever import BaseRetriever

class AutoMergingRetriever(BaseRetriever):
    """
    Retriever that automatically merges nodes into parent nodes when appropriate.
    """
    
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        storage_context: StorageContext,
        simple_ratio_thresh: float = 0.5,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the auto-merging retriever.
        
        Args:
            vector_retriever: Base vector retriever
            storage_context: Storage context with docstore
            simple_ratio_thresh: Threshold for node ratio to merge
            verbose: Whether to print verbose output
        """
        super().__init__(**kwargs)
        self._vector_retriever = vector_retriever
        self._storage_context = storage_context
        self._simple_ratio_thresh = simple_ratio_thresh
        self._verbose = verbose
    
    def _get_parents_and_merge(self, nodes: List[NodeWithScore]) -> Tuple[List[NodeWithScore], bool]:
        """
        Get parents and merge nodes.
        
        Args:
            nodes: List of nodes to check for merging
            
        Returns:
            Tuple of (merged_nodes, is_changed)
        """
        # Retrieve all parent nodes
        # parent_nodes: Dict[str, BaseNode] = {}
        # parent_cur_children_dict: Dict[str, List[NodeWithScore]] = defaultdict(list)
        
        # for node in nodes:
        #     if node.node.parent_node is None:
        #         continue
            
        #     parent_node_info = node.node.parent_node
            
        #     # Fetch actual parent node if doesn't exist in the cache yet
        #     parent_node_id = parent_node_info.get("node_id")
        #     if parent_node_id not in parent_nodes:
        #         parent_node = self._storage_context.docstore.get_document(parent_node_id)
        #         parent_nodes[parent_node_id] = cast(BaseNode, parent_node)
            
        #     # Add reference to child from parent
        #     parent_cur_children_dict[parent_node_id].append(node)
        
        # # Compute ratios and "merge" nodes
        # node_ids_to_delete = set()
        # nodes_to_add: Dict[str, NodeWithScore] = {}
        
        # for parent_node_id, parent_node in parent_nodes.items():
        #     parent_child_nodes = parent_node.child_nodes
        #     parent_num_children = len(parent_child_nodes) if parent_child_nodes else 1
        #     parent_cur_children = parent_cur_children_dict[parent_node_id]
        #     ratio = len(parent_cur_children) / parent_num_children
            
        #     # If ratio is high enough, merge
        #     if ratio > self._simple_ratio_thresh:
        #         node_ids_to_delete.update(set(n.node.node_id for n in parent_cur_children))
                
        #         parent_node_text = truncate_text(
        #             parent_node.get_content(metadata_mode=MetadataMode.NONE), 100
        #         )
        #         info_str = (
        #             f"> Merging {len(parent_cur_children)} nodes into parent node.\n"
        #             f"> Parent node id: {parent_node_id}.\n"
        #             f"> Parent node text: {parent_node_text}\n"
        #         )
                
        #         self.logger.info(info_str)
        #         if self._verbose:
        #             print(info_str)
                
        #         # Add parent node
        #         # Averaging score across embeddings
        #         avg_score = sum(
        #             [n.get_score() or 0.0 for n in parent_cur_children]
        #         ) / len(parent_cur_children)
                
        #         parent_node_with_score = NodeWithScore(node=parent_node, score=avg_score)
        #         nodes_to_add[parent_node_id] = parent_node_with_score
        
        # # Delete old child nodes, add new parent nodes
        # new_nodes = [n for n in nodes if n.node.node_id not in node_ids_to_delete]
        # # Add parent nodes
        # new_nodes.extend(list(nodes_to_add.values()))
        
        # is_changed = len(node_ids_to_delete) > 0
        
        # return new_nodes, is_changed
        pass
    
    def _fill_in_nodes(self, nodes: List[NodeWithScore]) -> Tuple[List[NodeWithScore], bool]:
        """
        Fill in intermediate nodes.
        
        Args:
            nodes: List of nodes to check for filling
            
        Returns:
            Tuple of (filled_nodes, is_changed)
        """
        new_nodes = []
        is_changed = False
        
        for idx, node in enumerate(nodes):
            new_nodes.append(node)
            if idx >= len(nodes) - 1:
                continue
            
            cur_node = cast(BaseNode, node.node)
            # If there's a node in the middle, add that to the queue
            next_node_info = getattr(cur_node, "next_node", None)
            next_idx_node_info = getattr(nodes[idx + 1].node, "prev_node", None)
            
            if (next_node_info is not None and 
                next_idx_node_info is not None and 
                next_node_info.get("node_id") == next_idx_node_info.get("node_id")):
                
                is_changed = True
                next_node_id = next_node_info.get("node_id")
                next_node = self._storage_context.docstore.get_document(next_node_id)
                next_node = cast(BaseNode, next_node)
                
                next_node_text = truncate_text(
                    next_node.get_content(metadata_mode=MetadataMode.NONE), 100
                )
                
                info_str = (
                    f"> Filling in node. Node id: {next_node_id}"
                    f"> Node text: {next_node_text}\n"
                )
                
                self.logger.info(info_str)
                if self._verbose:
                    print(info_str)
                
                # Set score to be average of current node and next node
                avg_score = (node.get_score() + nodes[idx + 1].get_score()) / 2
                new_nodes.append(NodeWithScore(node=next_node, score=avg_score))
        
        return new_nodes, is_changed
    
    def _try_merging(self, nodes: List[NodeWithScore]) -> Tuple[List[NodeWithScore], bool]:
        """
        Try different ways to merge nodes.
        
        Args:
            nodes: List of nodes to merge
            
        Returns:
            Tuple of (merged_nodes, is_changed)
        """
        # First try filling in nodes
        nodes, is_changed_0 = self._fill_in_nodes(nodes)
        # Then try merging nodes
        nodes, is_changed_1 = self._get_parents_and_merge(nodes)
        
        return nodes, is_changed_0 or is_changed_1
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes given query.
        
        Args:
            query_bundle: Query bundle containing the query string
            
        Returns:
            List of retrieved nodes with scores
        """
        initial_nodes = self._vector_retriever.retrieve(query_bundle)
        
        cur_nodes, is_changed = self._try_merging(initial_nodes)
        
        while is_changed:
            cur_nodes, is_changed = self._try_merging(cur_nodes)
        
        # Sort by similarity
        cur_nodes.sort(key=lambda x: x.get_score(), reverse=True)
        
        return cur_nodes