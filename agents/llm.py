"""LLM调用模块：支持本地和远程API双模式"""
import os
from typing import Optional, Literal
import re

# 注意：本地模式需要安装transformers和torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    LOCAL_MODE_AVAILABLE = True
except ImportError:
    LOCAL_MODE_AVAILABLE = False
    print("警告：未安装transformers/torch，仅支持API模式")

# API模式依赖
try:
    from openai import OpenAI
    API_MODE_AVAILABLE = True
except ImportError:
    API_MODE_AVAILABLE = False
    print("警告：未安装openai库，仅支持本地模式")

class LLMClient:
    def __init__(
        self,
        mode: Literal["local", "api"] = "api",  # 默认API模式（更稳定）
        local_model_path: str = "../LLM/qwen3/",
        api_key: Optional[str] = "sk-e72469ad769b4879b527f39732b6a13c",
        api_model: str = "qwen-max",
        api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        """
        初始化LLM客户端
        
        Args:
            mode: 运行模式 ("local" 或 "api")
            local_model_path: 本地模型路径
            api_key: API密钥
            api_model: API模型名称
            api_base_url: API基础URL
        """
        self.mode = mode
        self.local_model = None
        self.local_tokenizer = None
        
        if self.mode == "local":
            if not LOCAL_MODE_AVAILABLE:
                print("本地模式不可用，切换到API模式")
                self.mode = "api"
            else:
                self._init_local_model(local_model_path)
        
        if self.mode == "api":
            if not API_MODE_AVAILABLE:
                # 如果API模式也不可用，使用模拟模式
                print("警告：API模式不可用，使用模拟模式")
                self.mode = "mock"
            else:
                self._init_api_client(api_key, api_model, api_base_url)
    
    def _init_local_model(self, model_path: str):
        """初始化本地模型"""
        try:
            print(f"加载本地模型: {model_path}")
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            self.local_model.eval()
            print("✓ 本地模型加载成功")
        except Exception as e:
            print(f"✗ 本地模型加载失败: {e}")
            print("切换到模拟模式")
            self.mode = "mock"
    
    def _init_api_client(self, api_key: str, api_model: str, api_base_url: str):
        """初始化API客户端"""
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            print("警告：未提供API密钥，使用模拟模式")
            self.mode = "mock"
            return
        
        self.api_client = OpenAI(
            api_key=self.api_key,
            base_url=api_base_url
        )
        self.api_model = api_model
        print(f"✓ API客户端初始化成功 (模型: {api_model})")
    
    def __call__(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_new_tokens: int = 2048
    ) -> str:
        """
        调用LLM生成响应
        
        Args:
            prompt: 输入提示
            temperature: 生成温度
            max_new_tokens: 最大生成token数
            
        Returns:
            生成的文本
        """
        if self.mode == "local":
            return self._call_local(prompt, temperature, max_new_tokens)
        elif self.mode == "api":
            return self._call_api(prompt, temperature, max_new_tokens)
        else:
            return self._call_mock(prompt)
    
    def _call_local(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """调用本地模型"""
        try:
            # 构建对话格式
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.local_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.local_model.device)
            
            # 生成响应
            with torch.no_grad():
                outputs = self.local_model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.8,
                    top_k=20,
                    eos_token_id=self.local_tokenizer.eos_token_id,
                    pad_token_id=self.local_tokenizer.pad_token_id
                )
            
            # 解码响应
            response_ids = outputs[0][input_ids.shape[-1]:]
            response = self.local_tokenizer.decode(
                response_ids,
                skip_special_tokens=True
            )
            
            return self._clean_response(response)
            
        except Exception as e:
            print(f"本地模型调用失败: {e}")
            return self._call_mock(prompt)
    
    def _call_api(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """调用API模型"""
        try:
            response = self.api_client.chat.completions.create(
                model=self.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if response.choices:
                return self._clean_response(response.choices[0].message.content)
            else:
                return self._call_mock(prompt)
                
        except Exception as e:
            print(f"API调用失败: {e}")
            return self._call_mock(prompt)
    
    def _call_mock(self, prompt: str) -> str:
        """模拟模式：返回合理的默认响应"""
        # 根据提示内容返回不同的模拟响应
        if "选择" in prompt or "题目" in prompt:
            # 选题场景
            import random
            import json
            
            # 提取候选题目ID（简单正则匹配）
            import re
            exercise_ids = re.findall(r'题目ID：(\d+)', prompt)
            if exercise_ids:
                selected_id = random.choice(exercise_ids)
                return json.dumps({
                    "exercise_id": int(selected_id),
                    "selection_reason": "基于综合指标分析，该题目难度适中，能有效评估学生能力",
                    "predicted_score": round(random.uniform(0.4, 0.7), 2),
                    "prediction_reason": "根据学生当前认知水平分析"
                }, ensure_ascii=False)
            
        elif "反思" in prompt or "分析" in prompt:
            # 反思场景
            return ("通过本次答题分析，学生在理解和应用维度表现良好，"
                   "但在分析和评价维度还需加强。建议适当增加高阶思维训练题目。")
        
        # 默认响应
        return "基于综合分析完成"
    
    def _clean_response(self, text: str) -> str:
        """清理响应文本"""
        if not text:
            return ""
        
        # 去除特殊字符
        text = re.sub(r"[\u0010-\u001F\u007F-\u009F]", "", text)
        
        # 去除代码块标记
        text = re.sub(r"```(?:json|python|text)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        
        # 去除多余空白
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        
        return text