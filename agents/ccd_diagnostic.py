import numpy as np
import json
import os
from bloom_taxonomy import BloomLevel, get_bloom_hierarchy

class CCDDiagnostic:
    """基于内容-认知维度(CCD)的诊断工具"""
    
    def __init__(self, knowledge_points, ccd_matrix_path=None):
        """初始化CCD诊断工具"""
        self.knowledge_points = knowledge_points
        self.kp_index = {kp: i for i, kp in enumerate(knowledge_points)}
        self.bloom_levels = [level for level in BloomLevel]
        self.bl_index = {level: i for i, level in enumerate(self.bloom_levels)}
        
        # 初始化CCD矩阵 (知识点 x Bloom层次)，值范围[0,1]表示掌握程度
        self.ccd_matrix = np.zeros((len(knowledge_points), len(BloomLevel)))
        
        # 初始化学习率参数
        self.correct_increment = 0.15  # 答对时的增量
        self.incorrect_decrement = 0.1  # 答错时的减量
        
        if ccd_matrix_path and os.path.exists(ccd_matrix_path):
            self.load_ccd_matrix(ccd_matrix_path)
            print(f"成功加载CCD矩阵: {ccd_matrix_path}")
        else:
            print("初始化新的CCD矩阵")
    
    def load_ccd_matrix(self, file_path):
        """从文件加载CCD矩阵"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for kp, levels in data.items():
                if kp in self.kp_index:
                    kp_idx = self.kp_index[kp]
                    for level_str, value in levels.items():
                        level = BloomLevel.from_string(level_str)
                        if level and level in self.bl_index:
                            bl_idx = self.bl_index[level]
                            # 确保值在[0,1]范围内
                            self.ccd_matrix[kp_idx, bl_idx] = np.clip(float(value), 0, 1)
        except Exception as e:
            print(f"加载CCD矩阵失败: {str(e)}，将使用初始矩阵")
    
    def update_ccd_matrix(self, question, is_correct):
        """根据答题情况更新CCD矩阵"""
        kp = question['knowledge_point']
        bloom_level = question['bloom_level']
        
        if kp in self.kp_index and bloom_level in self.bl_index:
            kp_idx = self.kp_index[kp]
            bl_idx = self.bl_index[bloom_level]
            
            # 根据答题结果更新掌握程度(0-1)
            if is_correct:
                new_value = self.ccd_matrix[kp_idx, bl_idx] + self.correct_increment
            else:
                new_value = self.ccd_matrix[kp_idx, bl_idx] - self.incorrect_decrement
            
            # 确保值在[0,1]范围内
            self.ccd_matrix[kp_idx, bl_idx] = np.clip(new_value, 0, 1)
    
    def get_knowledge_mastery(self, knowledge_point, bloom_level=None):
        """获取学生在特定知识点和认知层次的掌握程度"""
        if knowledge_point not in self.kp_index:
            return 0.0
            
        kp_idx = self.kp_index[knowledge_point]
        
        if bloom_level:
            if bloom_level not in self.bl_index:
                return 0.0
            bl_idx = self.bl_index[bloom_level]
            return self.ccd_matrix[kp_idx, bl_idx]
        else:
            # 返回该知识点所有认知层次的平均掌握程度
            return np.mean(self.ccd_matrix[kp_idx, :])
    
    def diagnose_weaknesses(self, top_n=3):
        """诊断学生的薄弱环节"""
        weaknesses = []
        
        for i, kp in enumerate(self.knowledge_points):
            for j, bl in enumerate(self.bloom_levels):
                mastery = self.ccd_matrix[i, j]
                weaknesses.append({
                    'knowledge_point': kp,
                    'bloom_level': bl,
                    'mastery': float(mastery),
                    'hierarchy': get_bloom_hierarchy(bl)
                })
        
        # 按掌握程度升序排序，取前n个
        weaknesses.sort(key=lambda x: x['mastery'])
        return weaknesses[:top_n]
    
    def save_ccd_matrix(self, file_path):
        """保存CCD矩阵到文件"""
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            data = {}
            for i, kp in enumerate(self.knowledge_points):
                levels = {}
                for j, bl in enumerate(self.bloom_levels):
                    levels[bl.value] = float(self.ccd_matrix[i, j])
                data[kp] = levels
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(f"保存CCD矩阵失败: {str(e)}")
            return False
    
    def get_ccd_matrix(self):
        """获取完整的CCD矩阵（用于调试）"""
        return self.ccd_matrix.copy()
    