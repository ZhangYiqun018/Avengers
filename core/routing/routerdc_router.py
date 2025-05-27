from typing import Tuple, List

from loguru import logger

from core.experts.load_experts import Expert
from core.routing.base_router import BaseRouter, RouterOutput

from transformers import AutoTokenizer, AutoModel

import random
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
class RouterModule(nn.Module):
    """
    路由模型模块，用相似度或其他策略输出对各个路由节点（模型）的打分。
    """
    def __init__(
        self,
        backbone: nn.Module,
        embededLLM_dim: int,
        embededLLM_path: str,
        llm_labels: List,
        hidden_state_dim: int = 1024,
        node_size: int = 3,
        similarity_function: str = "cos",

    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.similarity_function = similarity_function

        self.embeddings = nn.Embedding(node_size, embededLLM_dim)
        self.init_embedding_with_EmbedLLM(embededLLM_dim, embededLLM_path, llm_labels)

        self.embeding_project_linear = nn.Linear(embededLLM_dim, hidden_state_dim)
        # self.similarity_mlp = nn.Sequential(
        #     nn.Linear(hidden_state_dim * (node_size + 1), hidden_state_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_state_dim, node_size)
        # )
        self.similarity_mlp = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_state_dim),
            nn.ReLU(),
            nn.Linear(hidden_state_dim, 1)
        )

    def init_embedding_with_EmbedLLM(self, embededLLM_dim, embededLLM_path, llm_labels):
        """
        根据 dataset_name 从预先保存的 .npy 文件中加载 embedding，用于初始化 self.embeddings.weight。
        """
        loaded_embeddings = np.load(embededLLM_path, allow_pickle=True).item()
        embedding_list = []
        for model_name in llm_labels:
            vector = loaded_embeddings[model_name]
            embedding_list.append(vector)

        embedding_tensor = torch.tensor(embedding_list, dtype=torch.float32)
        self.embeddings.weight = nn.Parameter(embedding_tensor)
        self.embeddings.weight.requires_grad = False

    def compute_similarity(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        计算相似度，默认为 cos，相乘后除以各自 norm。
        """
        if self.similarity_function == "cos":
            norm1 = torch.norm(input1, dim=1).unsqueeze(1)
            norm2 = torch.norm(input2, dim=1).unsqueeze(0)
            similarity = (input1 @ input2.T) / (norm1 * norm2)
        else:
            similarity = input1 @ input2.T
        return similarity

    def forward(self, t=1, **input_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回路由打分（logits）和隐藏向量。
        t: 温度系数（可用于缩放 logits）。
        """
        x = self.backbone(**input_kwargs)
        hidden_state = x['last_hidden_state'][:, 0, :]  # (batch_size, hidden_dim)
        

        '''
        x_expanded = hidden_state.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        y_expanded = y.unsqueeze(0).expand(hidden_state.size(0), -1, -1)  # (batch_size, node_size, hidden_dim)
        concatenated = torch.cat([x_expanded, y_expanded], dim=1)
        concatenated = concatenated.reshape(hidden_state.size(0), -1)  # (batch_size, (node_size+1)*hidden_dim)
        out = self.similarity_mlp(concatenated)
        '''
   
        x = hidden_state # 这里的x维度是(batch_size, hidden_dim)
        y = self.embeding_project_linear(self.embeddings.weight) 
        # 这里y的维度是(llm_num, hidden_dim)
        # 我希望你把 batch里每一条x 和 每一个llm的embedding按位乘，然后得到mul_xy

        x_expanded = x.unsqueeze(1) 
        y_expanded = y.unsqueeze(0) 
        mul_xy = x_expanded * y_expanded
        out = self.similarity_mlp(mul_xy)

        logits = out.squeeze(-1)
        logits = logits / t
        return logits, hidden_state
        

class RouterDC(BaseRouter):
    def __init__(self, normal_experts: List[Expert], thinking_experts: List[Expert], router_config: dict):
        super().__init__(normal_experts, thinking_experts)
        
        self.config = router_config['routerdc_router']
        self.encoder_model_path = self.config['encoder_model_path']
        self.model_path = self.config['model_path']
        self.max_router = self.config['max_router']

        # setup_seed(config.seed)
        MODEL_NAME = "/fs-computility/mabasic/shared/models/jina-embeddings-v3"
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            truncation_side='left',
            padding=True,
            trust_remote_code=True
        )
        backbone_model = AutoModel.from_pretrained(self.encoder_model_path, trust_remote_code=True)
        node_size = len( self.config["llm_labels"])
        self.router_model = RouterModule(
            backbone=backbone_model,
            hidden_state_dim=1024,       
            node_size=node_size,                     
            similarity_function=self.config["similarity_function"], 
            embededLLM_dim=self.config["embededLLM_dim"],           
            embededLLM_path=self.config["embededLLM_path"],         
            llm_labels=self.config["llm_labels"]        
        )
        
        logger.info(f"Loading model weights from {self.model_path}")
        state_dict = torch.load(self.model_path, map_location='cpu')
        self.router_model.load_state_dict(state_dict)
        self.router_model.to(device)
    
    def route(self, question: str) -> RouterOutput:
        # Add lock for thread safety during model inference
        if not hasattr(self, '_lock'):
            import threading
            self._lock = threading.Lock()
        with self._lock:
            inputs = self.tokenizer(
                question,
                return_tensors='pt',
                truncation=True,
                max_length=1024
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits, _ = self.router_model(**inputs)
            weights = torch.sigmoid(logits).to(device)
        
        sorted_weights, indices = torch.sort(weights, descending=True)
        top_k_indices = indices[:self.max_router].cpu().numpy()
        
        # Select normal experts based on top k indices
        selected_normal_experts = []
        for idx in top_k_indices[0]:
            idx = int(idx)
            if idx == 0:
                expert = next((e for e in self.normal_experts if "bio" in e.model_name.lower()), None)
            elif idx == 1:
                expert = next((e for e in self.normal_experts if "eurus" in e.model_name.lower()), None)
            elif idx == 2:
                expert = next((e for e in self.normal_experts if "fin-r1" in e.model_name.lower()), None)
            elif idx == 3:
                expert = next((e for e in self.normal_experts if "glm-4-9b-chat" in e.model_name.lower()), None)
            elif idx == 4:
                expert = next((e for e in self.normal_experts if "qwen2.5-7b-instruct" in e.model_name.lower()), None)
            elif idx == 5:
                expert = next((e for e in self.normal_experts if "qwen2.5-coder-7b-instruct" in e.model_name.lower()), None)
            elif idx == 6:
                expert = next((e for e in self.normal_experts if "qwen2.5-math-7b-instruct" in e.model_name.lower()), None)
            else:
                raise ValueError(f"Invalid index: {idx}")
                
            if expert is None:
                raise ValueError(f"Could not find expert for index {idx}")
            selected_normal_experts.append(expert)
            
        return RouterOutput(
            normal_experts=selected_normal_experts[:self.max_router],
            # thinking experts con be empty or random experts
            thinking_experts=self.thinking_experts
        )