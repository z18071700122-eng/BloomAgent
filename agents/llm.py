import os
import dashscope
from dashscope import Generation
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

class QwenLLM:
    """使用DashScope官方SDK调用千问模型"""
    def __init__(self, 
            model_name="qwen3-max", 
            temperature=0.3,           
            max_tokens=8192,           # 新增：最大输出
            enable_thinking=True,      # 新增：开启深度思考
            api_key="sk-e72469ad769b4879b527f39732b6a13c"):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.api_key = api_key


        dashscope.api_key = self.api_key
        print(f"✓ 已初始化千问API模型: {model_name}")


    def __call__(self, prompt, max_tokens=None):
        """调用千问模型生成回答"""
        if max_tokens is None:
                max_tokens = self.max_tokens

        try:
            # 构建对话消息
            messages = [
                {"role": "system", "content": "你是一位教育评估领域的专家，擅长根据学生学习情况提供精准的题目推荐。"},
                {"role": "user", "content": prompt}
            ]
            
            # 调用DashScope API
            response = Generation.call(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                enable_thinking=self.enable_thinking,  # 新增：深度思考
                result_format="message"
            )
            
            # 处理响应
            if response.status_code == 200:
                return response.output.choices[0].message.content.strip()
            else:
                print(f"API调用失败: {response.status_code}, 错误信息: {response.message}")
                return "推荐失败，请检查API配置"
                
        except Exception as e:
            print(f"模型调用出错: {str(e)}")
            return "推荐失败，发生异常"
    

class LocalQwenLLM:
    """本地部署的Qwen3-8B模型调用"""
    def __init__(self, model_path="../LLM/Qwen3-8B/", temperature=0.3,max_tokens=8192):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 加载tokenizer和模型
        print(f"正在加载本地模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        print("本地模型加载完成")
    
    def __call__(self, prompt, max_tokens=None):
        """调用本地模型生成回答"""
        if max_tokens is None:
            max_tokens = self.max_tokens
        try:
            # 构建对话消息
            messages = [
                {"role": "system", "content": "你是一位教育评估领域的专家，擅长根据学生学习情况提供精准的题目推荐。"},
                {"role": "user", "content": prompt}
            ]
            
            # 准备输入
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 生成回答
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # 解析结果（去除思考过程，只保留最终回答）
            try:
                # 查找思考结束标记
                index = len(output_ids) - output_ids[::-1].index(151668)  # 151668是Qwen3的思考结束标记
            except ValueError:
                index = 0
                
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return content
            
        except Exception as e:
            print(f"本地模型调用出错: {str(e)}")
            return "推荐失败，发生异常"