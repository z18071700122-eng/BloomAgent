"""选题Agent：基于信息增益和最近发展区的智能选题（优化版）"""
import random
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from data_loader import DataLoader

class SelectAgent:
    """优化版选题Agent：多目标智能选题"""
    
    def __init__(self, llm=None, oca_data_path=None):
        super().__init__()
        self.llm = llm
        if self.llm is None:
            raise ValueError("必须提供LLM实例")
        
        # 初始化数据加载器
        self.data_loader = DataLoader()
        self.exercises = self.data_loader.exercises
        self.oca_data = self.data_loader.load_oca_data(oca_data_path)
        
        # Bloom认知维度定义
        self.bloom_dimensions = [
            "记忆(Remember)：回忆和识别信息的能力",
            "理解(Understand)：解释和归纳信息的能力",
            "应用(Apply)：在具体情境中使用知识的能力",
            "分析(Analyze)：将复杂信息分解并理解关系的能力",
            "评价(Evaluate)：基于标准进行判断和评估的能力",
            "创造(Create)：整合信息产生新想法的能力"
        ]
        
        # 增强版教师设定
        self.teacher_setting = (
            "你是一位精通自适应测试和认知诊断的教育测评专家。\n"
            "你的任务是基于以下原则选择最优题目：\n"
            "1. 信息增益最大化：优先选择能最大程度减少认知能力评估不确定性的题目\n"
            "2. 最近发展区原则：选择认知差距在[-0.3, 0.1]范围内的题目\n"
            "3. 诊断精度优化：选择能有效区分学生能力层次的题目\n"
            "4. 知识覆盖均衡：适当覆盖不同知识点，避免重复测试\n\n"
            "核心概念提醒：\n"
            "- 认知差距 = SCA - OCA\n"
            "- 差距接近0表示难度适中，信息量最大\n"
            "- 差距< -0.3表示过难，> 0.1表示过易"
        )
        
        # 选题历史记录
        self.selection_history = {
            'selected_exercises': [],
            'covered_knowledge': set(),
            'selection_reasons': []
        }
    
    def get_candidate_questions(self, history_exercise_ids: List[int], 
                               student_id: int, limit: int = 15) -> List[int]:
        """
        智能筛选候选题目
        优先级：
        1. 未测试的知识点
        2. 测试次数少的知识点  
        3. 高信息增益的题目
        """
        # 转换已做题目ID
        history_ids = [int(id) for id in history_exercise_ids]
        
        # 获取学生的所有可用题目
        student_exercises = self.data_loader.get_student_test_exercises(student_id)
        if not student_exercises:
            print(f"警告：学生{student_id}无测试数据，使用全部题目池")
            available_ids = list(self.exercises.keys())
        else:
            available_ids = list(student_exercises.keys())
        
        # 排除已做过的题目
        candidate_pool = [eid for eid in available_ids if eid not in history_ids]
        
        if not candidate_pool:
            print("候选池为空：所有题目已完成")
            return []
        
        # 智能筛选策略
        candidates = self._intelligent_filtering(
            candidate_pool, 
            history_exercise_ids,
            limit
        )
        
        print(f"✓ 筛选出{len(candidates)}道候选题（总候选池{len(candidate_pool)}道）")
        return candidates
    
    def _intelligent_filtering(self, candidate_pool: List[int], 
                             history_ids: List[int], 
                             limit: int) -> List[int]:
        """智能筛选候选题"""
        # 统计已测试的知识点
        tested_knowledge = set()
        for eid in history_ids:
            if eid in self.exercises:
                tested_knowledge.update(self.exercises[eid]['knowledge_code'])
        
        # 对候选题评分
        scored_candidates = []
        for eid in candidate_pool:
            if eid not in self.exercises:
                continue
            
            exercise = self.exercises[eid]
            knowledge_codes = exercise['knowledge_code']
            
            # 计算评分
            score = 0.0
            
            # 1. 未测试知识点优先（权重0.4）
            untested = [k for k in knowledge_codes if k not in tested_knowledge]
            score += 0.4 * len(untested) / max(len(knowledge_codes), 1)
            
            # 2. 知识点测试次数少优先（权重0.3）
            test_counts = [history_ids.count(eid) for eid in history_ids 
                          if eid in self.exercises and 
                          any(k in self.exercises[eid]['knowledge_code'] for k in knowledge_codes)]
            avg_test_count = sum(test_counts) / max(len(test_counts), 1) if test_counts else 0
            score += 0.3 * (1 - min(avg_test_count / 5, 1))  # 归一化到[0,1]
            
            # 3. 题目多样性（权重0.2）
            if 'area' in exercise:
                area_bonus = 0.2 if exercise['area'] not in self._get_recent_areas(history_ids) else 0
                score += area_bonus
            
            # 4. 随机因素（权重0.1），避免过于确定性
            score += 0.1 * random.random()
            
            scored_candidates.append((eid, score))
        
        # 按分数排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前limit个
        return [eid for eid, _ in scored_candidates[:limit]]
    
    def _get_recent_areas(self, history_ids: List[int], recent_n: int = 5) -> set:
        """获取最近做过的题目领域"""
        recent_areas = set()
        for eid in history_ids[-recent_n:]:
            if eid in self.exercises:
                area = self.exercises[eid].get('area', '')
                if area:
                    recent_areas.add(area)
        return recent_areas
    
    def _calculate_exercise_metrics(self, exercise_id: int, student_sca: List,
                                   reflection_insights: Dict) -> Dict:
        """计算题目的各项指标"""
        if exercise_id not in self.exercises or exercise_id not in self.oca_data:
            return None
        
        exercise = self.exercises[exercise_id]
        oca_info = self.oca_data[exercise_id]
        knowledge_codes = oca_info['involved_knowledge_ids']
        ocas = oca_info['OCA_involved']
        
        metrics = {
            'exercise_id': exercise_id,
            'knowledge_codes': knowledge_codes,
            'information_gain': 0.0,
            'zpd_score': 0.0,
            'difficulty_match': 0.0,
            'diagnostic_value': 0.0,
            'dimension_comparisons': [],
            'overall_score': 0.0
        }
        
        # 计算每个知识点的指标
        for i, kid in enumerate(knowledge_codes):
            if kid - 1 >= len(student_sca) or i >= len(ocas):
                continue
            
            sca_vector = np.array(student_sca[kid - 1])
            oca_vector = np.array(ocas[i])
            
            # 1. 信息增益
            info_gain = self.data_loader.calculate_information_gain(
                student_sca, ocas, [kid]
            )
            metrics['information_gain'] += info_gain
            
            # 2. ZPD得分
            is_zpd, zpd_score = self.data_loader.is_in_zpd(
                student_sca, ocas, [kid]
            )
            metrics['zpd_score'] += zpd_score
            
            # 3. 难度匹配度
            gap = sca_vector - oca_vector
            mean_gap = np.mean(gap)
            if -0.2 <= mean_gap <= 0.0:
                difficulty_match = 1.0
            else:
                difficulty_match = np.exp(-abs(mean_gap + 0.1) ** 2)
            metrics['difficulty_match'] += difficulty_match
            
            # 4. 诊断价值（考虑薄弱维度）
            diagnostic_value = 0.0
            weakness_dims = reflection_insights.get('weakness_dimensions', [])
            for dim_idx in weakness_dims:
                if dim_idx < len(gap) and gap[dim_idx] < -0.2:
                    diagnostic_value += 0.2  # 能诊断薄弱维度
            metrics['diagnostic_value'] += diagnostic_value
            
            # 5. 维度对比详情
            dim_comparison = []
            for j in range(6):
                diff = float(sca_vector[j] - oca_vector[j])
                if diff >= 0.1:
                    status = "超过要求"
                elif diff >= -0.1:
                    status = "基本匹配"
                elif diff >= -0.3:
                    status = "略有不足"
                else:
                    status = "明显不足"
                
                dim_comparison.append({
                    'dimension': self.bloom_dimensions[j].split('：')[0],
                    'sca': float(sca_vector[j]),
                    'oca': float(oca_vector[j]),
                    'gap': diff,
                    'status': status
                })
            
            metrics['dimension_comparisons'].append({
                'knowledge': self.data_loader.get_knowledge_name(kid),
                'comparisons': dim_comparison
            })
        
        # 归一化指标
        n_knowledge = len(knowledge_codes)
        if n_knowledge > 0:
            metrics['information_gain'] /= n_knowledge
            metrics['zpd_score'] /= n_knowledge
            metrics['difficulty_match'] /= n_knowledge
            metrics['diagnostic_value'] /= n_knowledge
        
        # 计算综合得分
        weights = {
            'information_gain': 0.35,  # 信息增益最重要
            'zpd_score': 0.25,         # ZPD原则次之
            'difficulty_match': 0.2,    # 难度匹配
            'diagnostic_value': 0.2     # 诊断价值
        }
        
        metrics['overall_score'] = sum(
            metrics[key] * weight 
            for key, weight in weights.items()
        )
        
        return metrics
    
    def select_question(self, student_profile: Dict, history_log: Dict, 
                       latest_reflection: str, student_id: int,
                       reflection_insights: Optional[Dict] = None,
                       candidate_limit: int = 15) -> Tuple[Optional[int], 
                                                           Optional[Tuple[float, str]], 
                                                           Optional[str]]:
        """
        基于多目标优化选择最优题目
        返回: (题目ID, (预测得分, 预测理由), 选题理由)
        """
        # 获取候选题
        history_exercise_ids = history_log['exercise_id']
        candidate_ids = self.get_candidate_questions(
            history_exercise_ids, student_id, candidate_limit
        )
        
        if not candidate_ids:
            print("无可用候选题目")
            return None, None, None
        
        # 获取学生SCA
        student_sca = student_profile.get('sca', [])
        if not student_sca:
            print("学生SCA数据为空")
            return None, None, None
        
        # 如果没有提供反思洞察，使用默认值
        if reflection_insights is None:
            reflection_insights = {
                'weakness_dimensions': [],
                'strength_dimensions': [],
                'learning_rate_adjustment': 1.0,
                'confidence_level': 0.5
            }
        
        # 计算所有候选题的指标
        all_metrics = []
        for eid in candidate_ids:
            metrics = self._calculate_exercise_metrics(
                eid, student_sca, reflection_insights
            )
            if metrics:
                all_metrics.append(metrics)
        
        if not all_metrics:
            print("无有效候选题目指标")
            return None, None, None
        
        # 按综合得分排序
        all_metrics.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # 选择前10个传给LLM做最终决策
        top_candidates = all_metrics[:min(10, len(all_metrics))]
        
        # 构建LLM提示
        prompt = self._build_selection_prompt(
            top_candidates, 
            student_profile,
            latest_reflection,
            reflection_insights
        )
        
        # 调用LLM
        response = self.llm(prompt)
        response = self.data_loader.filter_llm_response(response)
        
        # 解析响应
        try:
            result = json.loads(response)
            selected_id = int(result.get('exercise_id', top_candidates[0]['exercise_id']))
            predicted_score = float(result.get('predicted_score', 0.5))
            prediction_reason = result.get('prediction_reason', '基于综合分析')
            selection_reason = result.get('selection_reason', '基于指标优化')
            
            # 记录选题历史
            self.selection_history['selected_exercises'].append(selected_id)
            self.selection_history['selection_reasons'].append(selection_reason)
            if selected_id in self.exercises:
                self.selection_history['covered_knowledge'].update(
                    self.exercises[selected_id]['knowledge_code']
                )
            
            return selected_id, (predicted_score, prediction_reason), selection_reason
            
        except Exception as e:
            print(f"解析LLM响应失败: {e}，使用指标最优题目")
            # 使用指标最优的题目
            best = top_candidates[0]
            selected_id = best['exercise_id']
            
            # 基于指标预测
            if best['zpd_score'] > 0.7:
                predicted_score = 0.7  # 在ZPD内，预测较高正确率
            elif best['difficulty_match'] > 0.7:
                predicted_score = 0.6
            else:
                predicted_score = 0.4
            
            prediction_reason = f"基于指标分析（信息增益{best['information_gain']:.2f}）"
            selection_reason = f"综合得分最高（{best['overall_score']:.2f}）"
            
            return selected_id, (predicted_score, prediction_reason), selection_reason
    
    def _build_selection_prompt(self, candidates: List[Dict], 
                               student_profile: Dict,
                               latest_reflection: str,
                               reflection_insights: Dict) -> str:
        """构建选题提示"""
        prompt = f"{self.teacher_setting}\n\n"
        
        # 学生当前状态
        prompt += "【学生当前状态】\n"
        prompt += f"认知模式：{student_profile.get('cognitive_pattern', '发展中')}\n"
        prompt += f"学习风格：{student_profile.get('learning_style', '综合型')}\n"
        
        if 'statistics' in student_profile:
            stats = student_profile['statistics']
            prompt += f"答题统计：共{stats['total_questions']}题，正确率{stats['correct_rate']:.1%}\n"
            prompt += f"学习趋势：{stats['recent_trend']}\n"
        
        prompt += f"\n最新反思：{latest_reflection[:200]}\n"
        
        # 薄弱维度提醒
        if reflection_insights.get('weakness_dimensions'):
            bloom_names = ['记忆', '理解', '应用', '分析', '评价', '创造']
            weak_dims = [bloom_names[i] for i in reflection_insights['weakness_dimensions'] if i < 6]
            prompt += f"需重点关注维度：{', '.join(weak_dims)}\n"
        
        # 候选题目分析
        prompt += "\n【候选题目分析】\n"
        for idx, metrics in enumerate(candidates, 1):
            exercise_id = metrics['exercise_id']
            exercise_info = self.data_loader.get_exercise_info(exercise_id)
            
            prompt += f"\n{idx}. 题目ID：{exercise_id}\n"
            if exercise_info:
                prompt += f"   题目名称：{exercise_info['name']}\n"
                prompt += f"   知识点：{', '.join(exercise_info['knowledge_names'])}\n"
            
            # 核心指标
            prompt += f"   信息增益：{metrics['information_gain']:.3f}（越高表示能更好地评估能力）\n"
            prompt += f"   ZPD得分：{metrics['zpd_score']:.3f}（越高表示越在最近发展区内）\n"
            prompt += f"   难度匹配：{metrics['difficulty_match']:.3f}（越高表示难度越合适）\n"
            prompt += f"   诊断价值：{metrics['diagnostic_value']:.3f}（越高表示越能诊断薄弱点）\n"
            prompt += f"   综合评分：{metrics['overall_score']:.3f}\n"
            
            # 关键维度分析（只显示第一个知识点）
            if metrics['dimension_comparisons']:
                first_knowledge = metrics['dimension_comparisons'][0]
                prompt += f"   {first_knowledge['knowledge']}的认知对比：\n"
                critical_dims = [
                    comp for comp in first_knowledge['comparisons']
                    if comp['status'] in ['明显不足', '略有不足']
                ][:3]  # 只显示前3个关键维度
                for comp in critical_dims:
                    prompt += f"     - {comp['dimension']}：学生{comp['sca']:.2f} vs 题目{comp['oca']:.2f} ({comp['status']})\n"
        
        # 选题要求
        prompt += "\n【选题决策】\n"
        prompt += "请综合以上分析，选择1道最适合的题目。\n"
        prompt += "决策优先级：\n"
        prompt += "1. 优先选择信息增益高且在ZPD内的题目\n"
        prompt += "2. 适当考虑诊断薄弱维度的题目\n"
        prompt += "3. 避免选择过难(ZPD得分<0.3)或过易(难度匹配<0.3)的题目\n"
        
        prompt += "\n输出格式（JSON）：\n"
        prompt += "{\n"
        prompt += '  "exercise_id": 题目ID(整数),\n'
        prompt += '  "selection_reason": "选择该题的详细理由",\n'
        prompt += '  "predicted_score": 预测正确率(0.0-1.0),\n'
        prompt += '  "prediction_reason": "预测理由"\n'
        prompt += "}"
        
        return prompt
    
    def get_question_info(self, exercise_id: int) -> Optional[Dict]:
        """获取题目详细信息"""
        return self.data_loader.get_exercise_info(exercise_id)
    
    def get_selection_history(self) -> Dict:
        """获取选题历史"""
        return self.selection_history