"""诊断Agent：管理学生画像、答题记录并进行反思（优化版）"""
import os
import json
import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from data_loader import DataLoader

class DiagnoseAgent:
    """优化版诊断Agent：结合反思指导SCA更新"""
    
    def __init__(self, llm=None, sca_path=None, oca_path=None, 
                 mem_file_path="../results/memory/"):
        super().__init__()
        
        # 初始化存储路径
        self.mem_filepath = os.path.join(mem_file_path, "student_profile.json")
        os.makedirs(os.path.dirname(self.mem_filepath), exist_ok=True)
        
        # 初始化LLM
        self.llm = llm
        if self.llm is None:
            raise ValueError("必须提供LLM实例")
        
        # 初始化数据加载器
        self.data_loader = DataLoader()
        self.exercises = self.data_loader.exercises
        self.oca_data = self.data_loader.load_oca_data(oca_path)
        self.sca_data = self.data_loader.load_sca_data(sca_path)
        
        # 初始化内存存储
        self.memory = {
            'student_id': None,
            'latest_reflection': '',
            'reflection_insights': {  # 新增：反思洞察
                'weakness_dimensions': [],  # 薄弱维度
                'strength_dimensions': [],   # 优势维度
                'learning_rate_adjustment': 1.0,  # 学习率调整因子
                'confidence_level': 0.5  # 置信度
            },
            'history_log': {
                'exercise_id': [],
                'score': [],
                'knowledge_codes': [],
                'predicted_score': [],
                'prediction_reason': [],
                'history_reflections': [],
                'sca_updates': []  # 新增：记录SCA更新历史
            },
            'student_profile': {
                'sca': [],  # 完整SCA数据
                'mastery_level': {},  # 各知识点掌握水平
                'cognitive_pattern': '',  # 认知模式
                'learning_style': ''  # 学习风格
            }
        }
        
        # 教师角色设定（增强版）
        self.teacher_setting = (
            "你是一位精通Bloom认知分类法和认知诊断理论的资深教育专家。\n"
            "核心任务：基于学生答题表现，精准诊断其认知能力并提供可操作的改进建议。\n\n"
            "核心概念：\n"
            "- SCA（主观认知能力）：学生在特定知识点上的六维能力值（记忆→创造，递减排列）\n"
            "- OCA（客观认知属性）：题目对知识点的六维能力要求\n"
            "- 认知差距：SCA - OCA，正值表示能力超过要求，负值表示能力不足\n\n"
            "分析要求：\n"
            "1. 识别关键认知差距（特别是-0.3以下的严重不足维度）\n"
            "2. 分析预测偏差的深层原因\n"
            "3. 提出具体的能力提升建议\n"
            "4. 评估学生的学习潜力和发展方向"
        )
    
    def set_student_id(self, student_id: int):
        """设置学生ID并加载SCA数据"""
        student_id = int(student_id)
        self.memory['student_id'] = student_id
        
        if student_id in self.sca_data:
            # 加载完整SCA数据
            self.memory['student_profile']['sca'] = self.sca_data[student_id]
            
            # 计算初始掌握水平
            self._calculate_mastery_levels()
            
            # 分析认知模式
            self._analyze_cognitive_pattern()
            
            print(f"✓ 设置学生ID: {student_id}，加载SCA数据（{len(self.memory['student_profile']['sca'])}个知识点）")
        else:
            print(f"⚠ 警告：未找到学生 {student_id} 的SCA数据，将使用默认初始化")
            # 使用默认SCA初始化
            n_knowledge = 39  # 默认知识点数
            self.memory['student_profile']['sca'] = [
                [0.5, 0.45, 0.4, 0.35, 0.3, 0.25] for _ in range(n_knowledge)
            ]
    
    def _calculate_mastery_levels(self):
        """计算各知识点的掌握水平"""
        sca = self.memory['student_profile']['sca']
        mastery = {}
        
        for kid in range(len(sca)):
            # 综合六维能力计算掌握度
            weights = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]  # 低阶到高阶权重递减
            mastery_score = sum(sca[kid][i] * weights[i] for i in range(6))
            
            # 分级评定
            if mastery_score >= 0.7:
                level = "精通"
            elif mastery_score >= 0.5:
                level = "熟练"
            elif mastery_score >= 0.3:
                level = "基础"
            else:
                level = "薄弱"
            
            mastery[kid + 1] = {
                'score': mastery_score,
                'level': level
            }
        
        self.memory['student_profile']['mastery_level'] = mastery
    
    def _analyze_cognitive_pattern(self):
        """分析学生的认知模式"""
        sca = np.array(self.memory['student_profile']['sca'])
        
        # 计算各Bloom层次的平均表现
        bloom_avg = np.mean(sca, axis=0)
        
        # 识别认知模式
        if bloom_avg[0] > 0.7 and bloom_avg[2] < 0.4:
            pattern = "记忆型学习者：擅长记忆但应用能力不足"
        elif bloom_avg[3] > 0.6 and bloom_avg[5] > 0.4:
            pattern = "分析创新型：具备较强的高阶思维能力"
        elif np.std(bloom_avg) < 0.1:
            pattern = "均衡发展型：各认知层次发展较为均衡"
        elif bloom_avg[1] > bloom_avg[0] * 1.2:
            pattern = "理解导向型：注重理解但记忆基础薄弱"
        else:
            pattern = "发展中学习者：认知能力正在逐步提升"
        
        self.memory['student_profile']['cognitive_pattern'] = pattern
        
        # 识别学习风格
        if bloom_avg[2] > bloom_avg[1]:
            style = "实践型"
        elif bloom_avg[3] > bloom_avg[2]:
            style = "分析型"
        else:
            style = "理论型"
        
        self.memory['student_profile']['learning_style'] = style
    
    def update_sca_with_reflection(self, exercise_id: int, concept_id: int, 
                                  actual_score: float, predicted_score: float) -> np.ndarray:
        """
        结合反思结果更新SCA
        使用自适应学习率，根据预测准确度和反思洞察调整
        """
        if self.memory['student_id'] is None:
            return None
        
        # 获取当前SCA
        current_sca = np.array(self.memory['student_profile']['sca'][concept_id - 1])
        
        # 获取题目OCA
        if exercise_id in self.oca_data:
            oca_info = self.oca_data[exercise_id]
            # 找到对应知识点的OCA
            knowledge_codes = oca_info['involved_knowledge_ids']
            if concept_id in knowledge_codes:
                idx = knowledge_codes.index(concept_id)
                oca = np.array(oca_info['OCA_involved'][idx])
            else:
                print(f"警告：题目{exercise_id}的OCA中没有知识点{concept_id}")
                return current_sca
        else:
            print(f"警告：未找到题目{exercise_id}的OCA数据")
            return current_sca
        
        # 计算认知差距
        gap = current_sca - oca
        
        # 基础学习率
        base_lr = 0.1
        
        # 根据预测准确度调整学习率
        prediction_error = abs(predicted_score - actual_score)
        if prediction_error < 0.2:
            lr_multiplier = 0.8  # 预测准确，小幅调整
        elif prediction_error < 0.5:
            lr_multiplier = 1.0  # 中等误差，正常调整
        else:
            lr_multiplier = 1.5  # 预测偏差大，加大调整
        
        # 结合反思洞察调整学习率
        reflection_adjustment = self.memory['reflection_insights']['learning_rate_adjustment']
        confidence = self.memory['reflection_insights']['confidence_level']
        
        # 最终学习率
        learning_rate = base_lr * lr_multiplier * reflection_adjustment * (1.0 + (1.0 - confidence))
        
        # 更新策略
        if actual_score > 0.5:  # 答对
            # 提升低于OCA的维度
            for i in range(6):
                if gap[i] < 0:  # 能力不足的维度
                    # 根据差距大小调整提升幅度
                    boost = learning_rate * (1 - current_sca[i]) * abs(gap[i])
                    current_sca[i] = min(1.0, current_sca[i] + boost)
                elif gap[i] > 0.5:  # 能力远超要求
                    # 略微降低置信度（可能是猜对的）
                    current_sca[i] = max(0.0, current_sca[i] - learning_rate * 0.1)
        else:  # 答错
            # 降低高于OCA的维度
            for i in range(6):
                if gap[i] > 0:  # 原以为能力足够的维度
                    # 根据差距大小调整降低幅度
                    reduction = learning_rate * current_sca[i] * gap[i]
                    current_sca[i] = max(0.0, current_sca[i] - reduction)
                
                # 特别处理薄弱维度
                if i in self.memory['reflection_insights'].get('weakness_dimensions', []):
                    current_sca[i] = max(0.0, current_sca[i] - learning_rate * 0.05)
        
        # 确保递减性
        for i in range(5):
            if current_sca[i] < current_sca[i+1]:
                # 交换并微调
                current_sca[i], current_sca[i+1] = current_sca[i+1], current_sca[i]
                current_sca[i] = min(1.0, current_sca[i] + 0.01)
                current_sca[i+1] = max(0.0, current_sca[i+1] - 0.01)
        
        # 更新存储
        self.memory['student_profile']['sca'][concept_id - 1] = current_sca.tolist()
        
        # 记录更新历史
        self.memory['history_log']['sca_updates'].append({
            'step': len(self.memory['history_log']['exercise_id']),
            'concept_id': concept_id,
            'learning_rate': learning_rate,
            'old_sca': list(current_sca),
            'new_sca': current_sca.tolist()
        })
        
        return current_sca
    
    def update_history_log(self, exercise_id: int, score: float, 
                          knowledge_codes: List[int], predicted_score: float, 
                          prediction_reason: str):
        """更新答题历史并触发SCA更新"""
        try:
            exercise_id = int(exercise_id)
            knowledge_codes = [int(code) for code in knowledge_codes]
            
            # 保存当前反思到历史
            if self.memory['latest_reflection']:
                self.memory['history_log']['history_reflections'].append(
                    self.memory['latest_reflection']
                )
            
            # 更新历史记录
            self.memory['history_log']['exercise_id'].append(exercise_id)
            self.memory['history_log']['score'].append(score)
            self.memory['history_log']['knowledge_codes'].append(knowledge_codes)
            self.memory['history_log']['predicted_score'].append(predicted_score)
            self.memory['history_log']['prediction_reason'].append(prediction_reason)
            
            # 更新每个涉及知识点的SCA
            for kid in knowledge_codes:
                self.update_sca_with_reflection(exercise_id, kid, score, predicted_score)
            
            # 重新计算掌握水平
            self._calculate_mastery_levels()
            
            # 生成新的反思
            self.reflect()
            
        except Exception as e:
            print(f"更新历史记录出错: {str(e)}")
    
    def _extract_reflection_insights(self, reflection_text: str):
        """从反思文本中提取关键洞察"""
        insights = self.memory['reflection_insights']
        
        # 识别薄弱和优势维度
        weakness_keywords = ['不足', '薄弱', '较弱', '欠缺', '待提升']
        strength_keywords = ['较强', '优势', '擅长', '掌握良好']
        
        # Bloom维度映射
        bloom_dims = {
            '记忆': 0, '理解': 1, '应用': 2, 
            '分析': 3, '评价': 4, '创造': 5
        }
        
        # 重置维度列表
        insights['weakness_dimensions'] = []
        insights['strength_dimensions'] = []
        
        for dim_name, dim_idx in bloom_dims.items():
            if dim_name in reflection_text:
                # 检查该维度的评价
                for keyword in weakness_keywords:
                    if f"{dim_name}.*{keyword}" in reflection_text or f"{keyword}.*{dim_name}" in reflection_text:
                        insights['weakness_dimensions'].append(dim_idx)
                        break
                
                for keyword in strength_keywords:
                    if f"{dim_name}.*{keyword}" in reflection_text or f"{keyword}.*{dim_name}" in reflection_text:
                        insights['strength_dimensions'].append(dim_idx)
                        break
        
        # 调整学习率因子
        if '严重' in reflection_text or '明显不足' in reflection_text:
            insights['learning_rate_adjustment'] = 1.5
        elif '略微' in reflection_text or '稍有' in reflection_text:
            insights['learning_rate_adjustment'] = 0.8
        else:
            insights['learning_rate_adjustment'] = 1.0
        
        # 更新置信度
        if '可能' in reflection_text or '或许' in reflection_text:
            insights['confidence_level'] = 0.3
        elif '确定' in reflection_text or '明确' in reflection_text:
            insights['confidence_level'] = 0.8
        else:
            insights['confidence_level'] = 0.5
    
    def reflect(self):
        """基于最近答题进行深度反思"""
        if len(self.memory['history_log']['exercise_id']) == 0:
            self.memory['latest_reflection'] = "暂无答题记录"
            return
        
        # 获取最近记录
        last_idx = len(self.memory['history_log']['exercise_id']) - 1
        exercise_id = self.memory['history_log']['exercise_id'][last_idx]
        actual_score = self.memory['history_log']['score'][last_idx]
        predicted_score = self.memory['history_log']['predicted_score'][last_idx]
        prediction_reason = self.memory['history_log']['prediction_reason'][last_idx]
        knowledge_codes = self.memory['history_log']['knowledge_codes'][last_idx]
        
        # 获取题目信息
        exercise_info = self.data_loader.get_exercise_info(exercise_id)
        if not exercise_info:
            exercise_name = f"题目{exercise_id}"
            knowledge_names = [f"知识点{k}" for k in knowledge_codes]
        else:
            exercise_name = exercise_info['name']
            knowledge_names = exercise_info['knowledge_names']
        
        # 构建认知对比数据
        cognitive_comparisons = []
        for kid in knowledge_codes:
            kid_idx = kid - 1
            if 0 <= kid_idx < len(self.memory['student_profile']['sca']):
                student_sca = self.memory['student_profile']['sca'][kid_idx]
                
                # 获取OCA
                if exercise_id in self.oca_data:
                    oca_info = self.oca_data[exercise_id]
                    if kid in oca_info['involved_knowledge_ids']:
                        idx = oca_info['involved_knowledge_ids'].index(kid)
                        exercise_oca = oca_info['OCA_involved'][idx]
                    else:
                        exercise_oca = [0.5] * 6
                else:
                    exercise_oca = [0.5] * 6
                
                # 计算差距
                gaps = [s - o for s, o in zip(student_sca, exercise_oca)]
                
                cognitive_comparisons.append({
                    'knowledge': self.data_loader.get_knowledge_name(kid),
                    'sca': student_sca,
                    'oca': exercise_oca,
                    'gaps': gaps
                })
        
        # 构建反思提示
        prompt = f"{self.teacher_setting}\n\n"
        prompt += "【答题情况分析】\n"
        prompt += f"题目：{exercise_name} (ID: {exercise_id})\n"
        prompt += f"涉及知识点：{', '.join(knowledge_names)}\n"
        prompt += f"预测结果：{'正确' if predicted_score > 0.5 else '错误'}（{predicted_score:.2f}）\n"
        prompt += f"实际结果：{'正确' if actual_score > 0.5 else '错误'}\n"
        prompt += f"预测理由：{prediction_reason}\n\n"
        
        prompt += "【认知能力对比】\n"
        for comp in cognitive_comparisons:
            prompt += f"\n{comp['knowledge']}：\n"
            prompt += f"  学生SCA：[{', '.join(f'{v:.3f}' for v in comp['sca'])}]\n"
            prompt += f"  题目OCA：[{', '.join(f'{v:.3f}' for v in comp['oca'])}]\n"
            prompt += f"  认知差距：[{', '.join(f'{v:+.3f}' for v in comp['gaps'])}]\n"
            
            # 识别关键差距
            critical_gaps = [(i, g) for i, g in enumerate(comp['gaps']) if abs(g) > 0.3]
            if critical_gaps:
                bloom_names = ['记忆', '理解', '应用', '分析', '评价', '创造']
                prompt += "  关键差距：\n"
                for i, g in critical_gaps:
                    if g < 0:
                        prompt += f"    - {bloom_names[i]}维度严重不足（差距{g:.3f}）\n"
                    else:
                        prompt += f"    - {bloom_names[i]}维度显著超出（差距{g:.3f}）\n"
        
        # 添加历史趋势分析
        if len(self.memory['history_log']['score']) >= 3:
            recent_scores = self.memory['history_log']['score'][-3:]
            trend = "上升" if recent_scores[-1] > recent_scores[0] else "下降" if recent_scores[-1] < recent_scores[0] else "平稳"
            prompt += f"\n【学习趋势】最近3次答题趋势：{trend}\n"
        
        prompt += "\n【反思要求】\n"
        prompt += "1. 分析预测与实际结果差异的根本原因\n"
        prompt += "2. 识别学生在各Bloom维度上的优势和不足\n"
        prompt += "3. 评估学生的学习潜力和改进空间\n"
        prompt += "4. 提供具体可行的能力提升建议\n"
        prompt += "请提供简洁精准的反思（200字以内）。"
        
        # 调用LLM生成反思
        raw_reflection = self.llm(prompt)
        cleaned_reflection = self.data_loader.filter_llm_response(raw_reflection)
        
        # 提取关键洞察
        self._extract_reflection_insights(cleaned_reflection)
        
        # 保存反思
        self.memory['latest_reflection'] = cleaned_reflection
        print(f"[反思] {cleaned_reflection[:100]}...")
    
    def get_student_profile(self) -> Dict:
        """获取增强版学生画像"""
        profile = self.memory['student_profile'].copy()
        
        # 添加统计信息
        if self.memory['history_log']['score']:
            profile['statistics'] = {
                'total_questions': len(self.memory['history_log']['score']),
                'correct_rate': sum(self.memory['history_log']['score']) / len(self.memory['history_log']['score']),
                'recent_trend': self._calculate_trend(),
                'mastery_distribution': self._get_mastery_distribution()
            }
        
        return profile
    
    def _calculate_trend(self) -> str:
        """计算学习趋势"""
        if len(self.memory['history_log']['score']) < 5:
            return "数据不足"
        
        recent = self.memory['history_log']['score'][-5:]
        earlier = self.memory['history_log']['score'][-10:-5] if len(self.memory['history_log']['score']) >= 10 else []
        
        if not earlier:
            return "稳定"
        
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        
        if recent_avg > earlier_avg + 0.1:
            return "明显进步"
        elif recent_avg > earlier_avg:
            return "略有进步"
        elif recent_avg < earlier_avg - 0.1:
            return "需要关注"
        else:
            return "保持稳定"
    
    def _get_mastery_distribution(self) -> Dict:
        """获取掌握度分布"""
        mastery_levels = self.memory['student_profile'].get('mastery_level', {})
        distribution = {'精通': 0, '熟练': 0, '基础': 0, '薄弱': 0}
        
        for _, info in mastery_levels.items():
            level = info.get('level', '薄弱')
            distribution[level] += 1
        
        return distribution
    
    def get_history_log(self) -> Dict:
        """获取答题历史"""
        return self.memory['history_log']
    
    def get_latest_reflection(self) -> str:
        """获取最新反思"""
        return self.memory['latest_reflection']
    
    def get_reflection_insights(self) -> Dict:
        """获取反思洞察"""
        return self.memory['reflection_insights']
    
    def save_memory(self) -> Optional[str]:
        """保存学生记忆数据"""
        try:
            with open(self.mem_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2, default=str)
            return self.mem_filepath
        except Exception as e:
            print(f"保存学生画像失败: {e}")
            return None
    
    def get_student_id(self) -> Optional[int]:
        """获取当前学生ID"""
        return self.memory.get('student_id')