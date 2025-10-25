"""数据加载工具：统一处理各类数据加载逻辑（优化版）"""
import os
import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple

class DataLoader:
    """优化版数据加载器：修复文件名和数据结构问题"""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        
        # 数据存储
        self.exercises = {}  # 题目信息
        self.oca_data = {}  # OCA数据
        self.sca_data = {}  # SCA数据
        self.test_set = {}  # 测试集
        self.knowledge_info = {}  # 知识点信息
        
        # 加载所有数据
        self._load_knowledge_info()
        self._load_exercises()
        self._load_test_set()
        
    def _load_knowledge_info(self):
        """加载知识点信息"""
        try:
            filepath = os.path.join(self.data_dir, "knowledge_info.json")
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    knowledge_list = json.load(f)
                
                # 构建ID到名称的映射
                self.knowledge_info = {
                    item['knowledge_id']: item['knowledge_name'] 
                    for item in knowledge_list
                }
                print(f"成功加载知识点信息：{len(self.knowledge_info)}个知识点")
            else:
                print(f"警告：未找到knowledge_info.json")
        except Exception as e:
            print(f"加载知识点信息失败: {e}")
    
    def _load_exercises(self):
        """加载题目信息"""
        try:
            filepath = os.path.join(self.data_dir, "exercise_info.json")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"题目信息文件不存在: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                exercise_list = json.load(f)
            
            # 构建题目字典
            for item in exercise_list:
                exer_id = item['exer_id']
                self.exercises[exer_id] = {
                    'id': exer_id,
                    'name': item['exer_name'],
                    'knowledge_code': item['knowledge_code'],
                    'topic': item.get('topic', ''),
                    'area': item.get('area', '')
                }
            
            print(f"成功加载题目信息：{len(self.exercises)}道题")
        except Exception as e:
            print(f"加载题目信息失败: {e}")
    
    def _load_test_set(self):
        """加载测试集数据"""
        try:
            filepath = os.path.join(self.data_dir, "test_set.json")
            if not os.path.exists(filepath):
                print(f"警告：test_set.json不存在，将使用全部题目作为候选")
                return
            
            with open(filepath, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # 按学生组织数据
            self.test_set = {}
            for student in test_data:
                user_id = int(student['user_id'])
                # 构建题目ID到答案的映射
                exercise_answers = {}
                for log in student['logs']:
                    exer_id = int(log['exer_id'])
                    exercise_answers[exer_id] = {
                        'score': log['score'],
                        'knowledge_code': log['knowledge_code']
                    }
                
                self.test_set[user_id] = exercise_answers
            
            print(f"成功加载测试集：{len(self.test_set)}名学生")
        except Exception as e:
            print(f"加载测试集失败: {e}")
    
    def load_oca_data(self, oca_path: Optional[str] = None) -> Dict:
        """加载OCA数据（修复版）"""
        try:
            if not oca_path:
                oca_path = os.path.join(self.data_dir, "exercise_oca.json")
            
            if not os.path.exists(oca_path):
                # 尝试在oca子目录查找
                oca_dir = os.path.join(self.data_dir, "oca")
                if os.path.exists(oca_dir):
                    oca_files = [f for f in os.listdir(oca_dir) if f.endswith('.json')]
                    if oca_files:
                        oca_path = os.path.join(oca_dir, oca_files[0])
                    else:
                        raise FileNotFoundError("未找到OCA文件")
                else:
                    raise FileNotFoundError(f"OCA文件不存在: {oca_path}")
            
            with open(oca_path, 'r', encoding='utf-8') as f:
                oca_list = json.load(f)
            
            # 构建OCA字典，适配新格式
            self.oca_data = {}
            for item in oca_list:
                exer_id = int(item['exer_id'])
                knowledge_codes = item['knowledge_code']
                oca_vectors = item['oca']
                
                # 确保OCA向量数量与知识点数量匹配
                if len(oca_vectors) != len(knowledge_codes):
                    print(f"警告：题目{exer_id}的OCA向量数与知识点数不匹配")
                    continue
                
                self.oca_data[exer_id] = {
                    'involved_knowledge_ids': knowledge_codes,
                    'OCA_involved': oca_vectors  # 保持原有字段名以兼容
                }
            
            print(f"成功加载OCA数据：{len(self.oca_data)}道题目")
            return self.oca_data
            
        except Exception as e:
            print(f"加载OCA数据失败: {e}")
            return {}
    
    def load_sca_data(self, sca_path: Optional[str] = None) -> Dict:
        """加载SCA数据（修复版）"""
        try:
            if not sca_path:
                sca_path = os.path.join(self.data_dir, "student_sca.json")
            
            if not os.path.exists(sca_path):
                # 尝试在sca子目录查找
                sca_dir = os.path.join(self.data_dir, "sca")
                if os.path.exists(sca_dir):
                    sca_files = [f for f in os.listdir(sca_dir) if f.endswith('.json')]
                    if sca_files:
                        sca_path = os.path.join(sca_dir, sca_files[0])
                    else:
                        raise FileNotFoundError("未找到SCA文件")
                else:
                    raise FileNotFoundError(f"SCA文件不存在: {sca_path}")
            
            with open(sca_path, 'r', encoding='utf-8') as f:
                sca_list = json.load(f)
            
            # 构建SCA字典
            self.sca_data = {}
            for item in sca_list:
                user_id = int(item['user_id'])
                # sca是所有知识点的六维向量列表
                self.sca_data[user_id] = item['sca']
            
            print(f"成功加载SCA数据：{len(self.sca_data)}名学生")
            return self.sca_data
            
        except Exception as e:
            print(f"加载SCA数据失败: {e}")
            return {}
    
    def get_exercise_info(self, exer_id: int) -> Optional[Dict]:
        """获取题目详细信息"""
        if exer_id in self.exercises:
            exercise = self.exercises[exer_id]
            # 获取知识点名称
            knowledge_names = []
            for kid in exercise['knowledge_code']:
                kname = self.knowledge_info.get(kid, f"知识点{kid}")
                knowledge_names.append(kname)
            
            return {
                'id': exer_id,
                'name': exercise['name'],
                'knowledge_codes': exercise['knowledge_code'],
                'knowledge_names': knowledge_names,
                'topic': exercise.get('topic', ''),
                'area': exercise.get('area', '')
            }
        return None
    
    def get_student_test_exercises(self, user_id: int) -> Dict:
        """获取学生的测试题目及答案"""
        if user_id in self.test_set:
            return self.test_set[user_id]
        return {}
    
    def get_knowledge_name(self, knowledge_id: int) -> str:
        """获取知识点名称"""
        return self.knowledge_info.get(knowledge_id, f"知识点{knowledge_id}")
    
    @staticmethod
    def filter_llm_response(text: str) -> str:
        """过滤LLM响应中的多余格式"""
        if not text:
            return ""
        # 去除代码块标记
        text = re.sub(r"```(?:json|text|python)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
        # 去除多余空白
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # 去除注释
        text = re.sub(r"//.*?(?=[\n,}])", "", text)
        return text
    
    def calculate_information_gain(self, student_sca: np.ndarray, 
                                  exercise_oca: List[List[float]], 
                                  knowledge_codes: List[int]) -> float:
        """
        计算选择某道题目的信息增益
        基于认知差距的不确定性来估算
        """
        total_gain = 0.0
        
        for i, kid in enumerate(knowledge_codes):
            if kid - 1 < len(student_sca) and i < len(exercise_oca):
                sca_vector = np.array(student_sca[kid - 1])
                oca_vector = np.array(exercise_oca[i])
                
                # 计算认知差距
                gap = sca_vector - oca_vector
                
                # 差距越接近0，不确定性越大，信息增益越高
                # 使用高斯函数建模
                uncertainty = np.exp(-np.sum(gap ** 2) / 0.1)
                
                # 考虑各维度的方差，方差大的维度信息增益更高
                variance = np.var(gap)
                
                total_gain += uncertainty * (1 + variance)
        
        return total_gain / len(knowledge_codes) if knowledge_codes else 0.0
    
    def is_in_zpd(self, student_sca: np.ndarray, 
                   exercise_oca: List[List[float]], 
                   knowledge_codes: List[int],
                   zpd_lower: float = -0.3, 
                   zpd_upper: float = 0.1) -> Tuple[bool, float]:
        """
        判断题目是否在学生的最近发展区(ZPD)
        返回: (是否在ZPD, ZPD得分)
        """
        zpd_scores = []
        
        for i, kid in enumerate(knowledge_codes):
            if kid - 1 < len(student_sca) and i < len(exercise_oca):
                sca_vector = np.array(student_sca[kid - 1])
                oca_vector = np.array(exercise_oca[i])
                
                # 计算平均认知差距
                mean_gap = np.mean(sca_vector - oca_vector)
                
                # 判断是否在ZPD范围内
                if zpd_lower <= mean_gap <= zpd_upper:
                    # 越接近ZPD中心点，得分越高
                    optimal_gap = (zpd_lower + zpd_upper) / 2
                    zpd_score = 1.0 - abs(mean_gap - optimal_gap) / (zpd_upper - zpd_lower)
                    zpd_scores.append(zpd_score)
                else:
                    zpd_scores.append(0.0)
        
        avg_zpd_score = np.mean(zpd_scores) if zpd_scores else 0.0
        is_in_zpd = avg_zpd_score > 0.5
        
        return is_in_zpd, avg_zpd_score