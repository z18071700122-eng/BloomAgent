import torch
import json
import os
import numpy as np
from transformers import BertModel, BertTokenizer
#from transformers import AutoModel, AutoTokenizer
from bloom_taxonomy import BloomLevel
from ccd_diagnostic import CCDDiagnostic

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class DiagnoseAgent:
    """诊断Agent：负责学生画像和基于CCD的认知诊断"""
    
    def __init__(self, llm=None, tokenizer=None,
                 ccd_matrix_path=None,
                 mem_file_path="../results/latest/memory/",  # 默认路径（会被main.py传入的路径覆盖）
                 bert_path="../pretrained_models/bert_base_chinese",
                 chat_model="qwen3-max", 
                 q_text=True, reflection=True):
        super().__init__()
        
        # 初始化存储路径（接收外部传入的路径）
        self.mem_filepath = os.path.join(mem_file_path, "student_profile.json")
        # 确保目录存在
        os.makedirs(os.path.dirname(self.mem_filepath), exist_ok=True)
        
        # 初始化LLM
        self.llm = llm
        if self.llm is None:
            raise ValueError("必须提供LLM实例")
        
        # 初始化知识点和CCD工具
        self.knowledge_points = self._load_knowledge_points()
        self.ccd = CCDDiagnostic(self.knowledge_points, ccd_matrix_path)
        
        # 初始化内存存储
        self.cold_num = 5  # 冷启动阶段题目数量
        self.memory = {
            'student_id': None,
            'recommend_reflection': '',
            'history_log': {
                'question_id': [],
                'answer_bi': [],
                'knowledge_points': [],  # 复数形式，支持多个知识点
                'bloom_level': []
            },
            'history_log_text': [],
            'student_profile': {
                'learning_ability': '',
                'learning_preference': '',
                'knowledge_mastery': {},  # 知识点掌握情况
                'cognitive_level': {}     # 认知层次分布
            }
        }
        
        # 教师角色设定
        self.teacher_setting = (
            "你是一位经验丰富的教育评估专家，擅长根据学生的答题情况进行学习诊断。"
            "请基于Bloom认知分类法分析学生的学习状态和认知水平。"
        )
        
        # 加载BERT模型用于文本处理
        try:
            bert_path = "C:/Users/LL/Desktop/VSCode/BloomAgent/pretrained_models/bert_base_chinese"
            self.bert = BertModel.from_pretrained(bert_path).to(device)
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
            #self.bert = AutoModel.from_pretrained("bert-base-chinese").to(device)
            #self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            print(f"成功加载BERT模型: {bert_path}")
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            self.bert = None
            self.tokenizer = None
        
        self.QTEXT = q_text
        self.REFLECTION = reflection
        
        # 加载题目映射数据
        self._load_question_data()
    
    def _load_knowledge_points(self):
        """加载知识点列表（路径：data/processed/knowledge_points.json）"""
        try:
            data_path = os.path.abspath("../data/processed")
            with open(os.path.join(data_path, "knowledge_points.json"), 'r', encoding='utf-8') as f:
                knowledge_points = json.load(f)
            print(f"成功加载 {len(knowledge_points)} 个知识点")
            return knowledge_points
        except Exception as e:
            print(f"加载知识点失败: {e}")
            return []
    
    def _load_question_data(self):
        """加载题目相关数据（路径：data/processed/）"""
        try:
            data_path = os.path.abspath("../data/processed")
            
            with open(os.path.join(data_path, "q2k.json"), 'r', encoding='utf-8') as f:
                self.q2k = json.load(f)  # 题目到知识点列表的映射
            
            with open(os.path.join(data_path, "q2bloom.json"), 'r', encoding='utf-8') as f:
                self.q2bloom = json.load(f)  # 题目到Bloom层次的映射
            
            with open(os.path.join(data_path, "q2text.json"), 'r', encoding='utf-8') as f:
                self.q2text = json.load(f)  # 题目文本
            
            with open(os.path.join(data_path, "questions.json"), 'r', encoding='utf-8') as f:
                self.questions = json.load(f)  # 题目详细信息
            
            print(f"成功加载题目数据，共 {len(self.questions)} 道题目")
            
        except Exception as e:
            print(f"加载题目数据失败: {e}")
            # 初始化空数据结构避免后续错误
            self.q2k = {}
            self.q2bloom = {}
            self.q2text = {}
            self.questions = {}
    
    def set_student_id(self, student_id):
        """设置学生ID"""
        self.memory['student_id'] = student_id
        print(f"已设置学生ID: {student_id}")
    
    def clear_memory(self, cold_num=None):
        """清空记忆，保留冷启动数据"""
        cold = cold_num if cold_num is not None else self.cold_num
        self.memory = {
            'student_id': self.memory['student_id'],
            'recommend_reflection': '',
            'history_log': {
                'question_id': self.memory['history_log']['question_id'][:cold],
                'answer_bi': self.memory['history_log']['answer_bi'][:cold],
                'knowledge_points': self.memory['history_log']['knowledge_points'][:cold],
                'bloom_level': self.memory['history_log']['bloom_level'][:cold]
            },
            'history_log_text': self.memory['history_log_text'][:cold],
            'student_profile': self.memory['student_profile']
        }
        print(f"已清空记忆，保留前 {cold} 条冷启动数据")
    
    def update_history_log(self, question_id, answer_bi, select_reason="", predict_answer=""):
        """更新答题历史记录，支持多知识点"""
        try:
            question_id_str = str(question_id)
            # 获取题目相关信息
            knowledge_points = self.q2k.get(question_id_str, [])  # 列表形式
            bloom_level_str = self.q2bloom.get(question_id_str, "")
            bloom_level = BloomLevel.from_string(bloom_level_str) if bloom_level_str else None
            
            # 更新历史记录
            self.memory['history_log']['question_id'].append(question_id)
            self.memory['history_log']['answer_bi'].append(int(answer_bi))
            self.memory['history_log']['knowledge_points'].append(knowledge_points)
            self.memory['history_log']['bloom_level'].append(bloom_level)
            
            # 更新CCD矩阵（为每个知识点更新）
            if knowledge_points and bloom_level:
                for kp in knowledge_points:
                    question_info = {
                        'knowledge_point': kp,
                        'bloom_level': bloom_level
                    }
                    self.ccd.update_ccd_matrix(question_info, bool(answer_bi))
            
            # 更新文本日志
            item = {}
            if self.QTEXT:
                item['question'] = self.q2text.get(question_id_str, f"题目 {question_id}")
                item['answer'] = "正确" if int(answer_bi) == 1 else "错误"
                item['knowledge_points'] = knowledge_points  # 显示知识点
            if self.REFLECTION:
                item['select_reason'] = select_reason
                item['predict_answer'] = predict_answer
            self.memory['history_log_text'].append(item)
            
            # 更新学生画像
            self._update_student_profile()
            
        except Exception as e:
            print(f"更新历史记录出错: {str(e)}, question_id = {question_id}")
    
    def _update_student_profile(self):
        """更新学生画像，为每个知识点计算掌握度"""
        # 更新知识点掌握情况
        for kp in self.knowledge_points:
            self.memory['student_profile']['knowledge_mastery'][kp] = {
                'overall': float(self.ccd.get_knowledge_mastery(kp)),
                'details': {
                    level.value: float(self.ccd.get_knowledge_mastery(kp, level)) 
                    for level in BloomLevel
                }
            }
        
        # 分析学生的认知层次分布
        self._analyze_cognitive_distribution()
    
    def _analyze_cognitive_distribution(self):
        """分析学生在各认知层次的表现"""
        levels_count = {level: {'correct': 0, 'total': 0} for level in BloomLevel}
        
        for i in range(len(self.memory['history_log']['question_id'])):
            level = self.memory['history_log']['bloom_level'][i]
            answer = self.memory['history_log']['answer_bi'][i]
            
            if level:
                levels_count[level]['total'] += 1
                if answer == 1:
                    levels_count[level]['correct'] += 1
        
        # 计算各层次正确率
        for level, stats in levels_count.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
            else:
                accuracy = 0.0
            self.memory['student_profile']['cognitive_level'][level.value] = {
                'accuracy': accuracy,
                'count': stats['total']
            }
    
    def reflection(self):
        """基于历史记录进行反思，更新学生能力和偏好描述"""
        if len(self.memory['history_log_text']) > self.cold_num:
            # 分析学习能力
            ability_prompt = (
                f"{self.teacher_setting}请用一句话总结学生的学习能力，"
                f"基于以下学习记录：{str(self.memory['history_log_text'][-3:])}"
            )
            self.memory['student_profile']['learning_ability'] = self.llm(ability_prompt)
            
            # 分析学习偏好
            preference_prompt = (
                f"{self.teacher_setting}请用一句话总结学生的学习偏好，"
                f"基于以下学习记录：{str(self.memory['history_log_text'][-3:])}"
            )
            self.memory['student_profile']['learning_preference'] = self.llm(preference_prompt)
            
            # 分析推荐反思
            reflection_prompt = (
                f"{self.teacher_setting}请用一句话反思当前的题目推荐策略，"
                f"基于以下学习记录：{str(self.memory['history_log_text'][-3:])}"
            )
            self.memory['recommend_reflection'] = self.llm(reflection_prompt)
    
    def get_diagnosis(self):
        """获取学生诊断结果"""
        # 获取薄弱环节
        weaknesses = self.ccd.diagnose_weaknesses()
        
        # 整理诊断结果
        diagnosis = {
            'student_id': self.memory['student_id'],
            'overall_mastery': {
                kp: self.memory['student_profile']['knowledge_mastery'][kp]['overall']
                for kp in self.knowledge_points
            },
            'cognitive_levels': self.memory['student_profile']['cognitive_level'],
            'weaknesses': weaknesses,
            'learning_ability': self.memory['student_profile']['learning_ability'],
            'learning_preference': self.memory['student_profile']['learning_preference']
        }
        
        return diagnosis
    
    def get_student_profile(self):
        """获取学生画像"""
        return self.memory['student_profile']
    
    def get_history_text(self):
        """获取历史记录文本"""
        if self.QTEXT:
            return [
                (f"题目: {item['question']}", 
                 f"学生答案: {item['answer']}, 知识点: {item.get('knowledge_points', [])}") 
                for item in self.memory['history_log_text']
            ]
        else:
            return [
                (f"题目ID: {self.memory['history_log']['question_id'][i]}", 
                 f"答案: {self.memory['history_log']['answer_bi'][i]}") 
                for i in range(len(self.memory['history_log']['question_id']))
            ]
    
    def save_memory(self):
        """保存记忆到文件（路径已指向results下的独立文件夹）"""
        # 将BloomLevel枚举转换为字符串以便序列化
        for i in range(len(self.memory['history_log']['bloom_level'])):
            level = self.memory['history_log']['bloom_level'][i]
            if isinstance(level, BloomLevel):
                self.memory['history_log']['bloom_level'][i] = level.value
        
        with open(self.mem_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4)
        
        # 保存CCD矩阵到同一目录
        ccd_matrix_path = f"{self.mem_filepath}_ccd_matrix.json"
        self.ccd.save_ccd_matrix(ccd_matrix_path)
        
        return self.mem_filepath
    