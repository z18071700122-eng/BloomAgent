"""BloomAgent主程序：协调双Agent运行自适应测试（优化版）"""
import os
import json
import datetime
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time

# 导入优化后的组件
from llm import LLMClient
from agent_diagnose import DiagnoseAgent
from agent_select import SelectAgent
from data_loader import DataLoader

class BloomAgentCAT:
    """BloomAgent CAT系统主类"""
    
    def __init__(self, config: Dict):
        """
        初始化CAT系统
        config: 配置字典
        """
        self.config = config
        self.data_loader = DataLoader(config.get('data_dir', '../data'))
        
        # 初始化LLM
        self.llm = self._init_llm()
        
        # 初始化Agent
        self.diagnose_agent = None
        self.select_agent = None
        
        # 测试统计
        self.test_statistics = {
            'total_students': 0,
            'total_questions': 0,
            'total_predictions': 0,
            'correct_predictions': 0,
            'convergence_cases': 0,
            'avg_test_length': 0
        }
    
    def _init_llm(self) -> LLMClient:
        """初始化LLM客户端"""
        mode = self.config.get('llm_mode', 'local')
        
        if mode == 'local':
            model_path = self.config.get('local_model_path', '../LLM/qwen3/')
            return LLMClient(
                mode='local',
                local_model_path=model_path
            )
        else:
            api_key = self.config.get('api_key')
            return LLMClient(
                mode='api',
                api_key=api_key,
                api_model=self.config.get('api_model', 'qwen3-max')
            )
    
    def _init_agents(self, results_dir: str):
        """初始化诊断和选题Agent"""
        self.diagnose_agent = DiagnoseAgent(
            llm=self.llm,
            sca_path=self.config.get('sca_path'),
            oca_path=self.config.get('oca_path'),
            mem_file_path=os.path.join(results_dir, "memory")
        )
        
        self.select_agent = SelectAgent(
            llm=self.llm,
            oca_data_path=self.config.get('oca_path')
        )
    
    def run_adaptive_test(self, student_id: int, max_steps: int = 30,
                         convergence_threshold: float = 0.02) -> Dict:
        """
        对单个学生运行自适应测试
        
        Args:
            student_id: 学生ID
            max_steps: 最大测试步数
            convergence_threshold: SCA收敛阈值
            
        Returns:
            测试结果字典
        """
        print(f"\n{'='*60}")
        print(f"开始测试学生 {student_id}")
        print(f"{'='*60}")
        
        # 初始化学生
        self.diagnose_agent.set_student_id(student_id)
        
        # 获取学生的测试题目
        student_exercises = self.data_loader.get_student_test_exercises(student_id)
        if not student_exercises:
            print(f"警告：学生{student_id}无测试数据")
            return None
        
        # 测试记录
        test_results = {
            'student_id': student_id,
            'start_time': datetime.datetime.now().isoformat(),
            'test_history': [],
            'sca_history': [],
            'convergence': False,
            'final_sca': None,
            'final_profile': None
        }
        
        # SCA变化历史（用于收敛检测）
        sca_changes = []
        previous_sca = None
        
        # 主测试循环
        for step in range(max_steps):
            print(f"\n--- 第 {step+1}/{max_steps} 题 ---")
            
            # 1. 获取学生画像和反思
            student_profile = self.diagnose_agent.get_student_profile()
            history_log = self.diagnose_agent.get_history_log()
            latest_reflection = self.diagnose_agent.get_latest_reflection()
            reflection_insights = self.diagnose_agent.get_reflection_insights()
            
            # 2. 选择下一题
            start_time = time.time()
            exercise_id, prediction, selection_reason = self.select_agent.select_question(
                student_profile=student_profile,
                history_log=history_log,
                latest_reflection=latest_reflection,
                student_id=student_id,
                reflection_insights=reflection_insights,
                candidate_limit=15
            )
            selection_time = time.time() - start_time
            
            if not exercise_id:
                print("无法选择题目，结束测试")
                break
            
            # 3. 获取实际答题结果
            if exercise_id in student_exercises:
                answer_info = student_exercises[exercise_id]
                actual_score = answer_info['score']
                knowledge_codes = answer_info['knowledge_code']
            else:
                # 如果没有记录，随机生成（仅用于测试）
                actual_score = random.choice([0.0, 1.0])
                knowledge_codes = self.exercises[exercise_id]['knowledge_code'] if exercise_id in self.exercises else []
                print(f"⚠ 题目{exercise_id}无真实记录，使用随机结果")
            
            # 4. 记录预测准确率
            predicted_score, prediction_reason = prediction if prediction else (0.5, "")
            if predicted_score in [0.0, 1.0]:
                self.test_statistics['total_predictions'] += 1
                if predicted_score == actual_score:
                    self.test_statistics['correct_predictions'] += 1
            
            # 5. 更新诊断Agent
            self.diagnose_agent.update_history_log(
                exercise_id=exercise_id,
                score=actual_score,
                knowledge_codes=knowledge_codes,
                predicted_score=predicted_score,
                prediction_reason=prediction_reason
            )
            
            # 6. 记录测试步骤
            exercise_info = self.data_loader.get_exercise_info(exercise_id)
            test_results['test_history'].append({
                'step': step + 1,
                'exercise_id': exercise_id,
                'exercise_name': exercise_info['name'] if exercise_info else f"题目{exercise_id}",
                'knowledge_codes': knowledge_codes,
                'predicted_score': predicted_score,
                'actual_score': actual_score,
                'prediction_reason': prediction_reason,
                'selection_reason': selection_reason,
                'selection_time': selection_time,
                'reflection': latest_reflection
            })
            
            # 7. 记录SCA状态
            current_sca = np.array(student_profile['sca'])
            test_results['sca_history'].append(current_sca.tolist())
            
            # 8. 检查收敛
            if previous_sca is not None:
                sca_change = np.mean(np.abs(current_sca - previous_sca))
                sca_changes.append(sca_change)
                
                # 检查最近5次的平均变化
                if len(sca_changes) >= 5:
                    recent_avg_change = np.mean(sca_changes[-5:])
                    print(f"SCA平均变化: {recent_avg_change:.4f}")
                    
                    if recent_avg_change < convergence_threshold:
                        print(f"✓ SCA已收敛（阈值{convergence_threshold}），测试结束")
                        test_results['convergence'] = True
                        self.test_statistics['convergence_cases'] += 1
                        break
            
            previous_sca = current_sca
            
            # 9. 输出当前状态
            result_symbol = "✓" if actual_score > 0.5 else "✗"
            print(f"题目: {exercise_info['name'] if exercise_info else exercise_id}")
            print(f"结果: {result_symbol} (预测{predicted_score:.1f}, 实际{actual_score:.1f})")
            print(f"选题用时: {selection_time:.2f}秒")
        
        # 10. 保存最终状态
        final_profile = self.diagnose_agent.get_student_profile()
        test_results['final_sca'] = final_profile['sca']
        test_results['final_profile'] = {
            'cognitive_pattern': final_profile.get('cognitive_pattern', ''),
            'learning_style': final_profile.get('learning_style', ''),
            'mastery_distribution': final_profile.get('statistics', {}).get('mastery_distribution', {}),
            'recent_trend': final_profile.get('statistics', {}).get('recent_trend', '')
        }
        test_results['end_time'] = datetime.datetime.now().isoformat()
        test_results['total_questions'] = len(test_results['test_history'])
        
        # 11. 更新统计
        self.test_statistics['total_questions'] += test_results['total_questions']
        
        # 12. 保存学生记忆
        self.diagnose_agent.save_memory()
        
        return test_results
    
    def batch_test(self, student_ids: Optional[List[int]] = None, 
                  limit: Optional[int] = None,
                  max_steps: int = 30) -> List[Dict]:
        """
        批量测试多个学生
        
        Args:
            student_ids: 要测试的学生ID列表，None则测试所有
            limit: 限制测试学生数量
            max_steps: 每个学生最大测试步数
            
        Returns:
            所有学生的测试结果列表
        """
        # 确定要测试的学生
        if student_ids is None:
            student_ids = list(self.data_loader.test_set.keys())
        
        if limit and limit > 0:
            student_ids = student_ids[:limit]
        
        print(f"\n{'='*60}")
        print(f"批量测试：共{len(student_ids)}名学生")
        print(f"{'='*60}")
        
        all_results = []
        
        for idx, student_id in enumerate(tqdm(student_ids, desc="测试进度")):
            print(f"\n[{idx+1}/{len(student_ids)}] 学生{student_id}")
            
            result = self.run_adaptive_test(
                student_id=student_id,
                max_steps=max_steps,
                convergence_threshold=self.config.get('convergence_threshold', 0.02)
            )
            
            if result:
                all_results.append(result)
                self.test_statistics['total_students'] += 1
        
        # 计算统计指标
        if all_results:
            self.test_statistics['avg_test_length'] = (
                self.test_statistics['total_questions'] / 
                self.test_statistics['total_students']
            )
        
        return all_results
    
    def generate_report(self, test_results: List[Dict], save_dir: str) -> str:
        """生成测试报告"""
        report = []
        report.append("="*70)
        report.append("BloomAgent CAT 测试报告")
        report.append("="*70)
        report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试配置:")
        report.append(f"  - LLM模式: {self.config.get('llm_mode', 'local')}")
        report.append(f"  - 最大题目数: {self.config.get('max_steps', 30)}")
        report.append(f"  - 收敛阈值: {self.config.get('convergence_threshold', 0.02)}")
        report.append("")
        
        # 整体统计
        report.append("【整体统计】")
        report.append(f"测试学生数: {self.test_statistics['total_students']}")
        report.append(f"总答题数: {self.test_statistics['total_questions']}")
        report.append(f"平均答题数: {self.test_statistics['avg_test_length']:.1f}")
        
        if self.test_statistics['total_predictions'] > 0:
            accuracy = (self.test_statistics['correct_predictions'] / 
                       self.test_statistics['total_predictions'])
            report.append(f"预测准确率: {accuracy:.2%} "
                         f"({self.test_statistics['correct_predictions']}/"
                         f"{self.test_statistics['total_predictions']})")
        
        report.append(f"收敛案例数: {self.test_statistics['convergence_cases']}")
        report.append("")
        
        # 个体分析
        if test_results:
            report.append("【个体分析】")
            
            # 计算各项指标
            test_lengths = [r['total_questions'] for r in test_results]
            convergence_rates = [r['convergence'] for r in test_results]
            
            report.append(f"答题数分布:")
            report.append(f"  - 最少: {min(test_lengths)}")
            report.append(f"  - 最多: {max(test_lengths)}")
            report.append(f"  - 平均: {np.mean(test_lengths):.1f}")
            report.append(f"  - 标准差: {np.std(test_lengths):.2f}")
            report.append(f"收敛比例: {sum(convergence_rates)/len(convergence_rates):.1%}")
            report.append("")
            
            # 认知模式分布
            cognitive_patterns = [r['final_profile'].get('cognitive_pattern', '未知') 
                                for r in test_results if r.get('final_profile')]
            if cognitive_patterns:
                report.append("认知模式分布:")
                pattern_counts = {}
                for pattern in cognitive_patterns:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                for pattern, count in sorted(pattern_counts.items(), 
                                            key=lambda x: x[1], reverse=True):
                    report.append(f"  - {pattern}: {count} ({count/len(cognitive_patterns):.1%})")
                report.append("")
            
            # 学习风格分布
            learning_styles = [r['final_profile'].get('learning_style', '未知')
                             for r in test_results if r.get('final_profile')]
            if learning_styles:
                report.append("学习风格分布:")
                style_counts = {}
                for style in learning_styles:
                    style_counts[style] = style_counts.get(style, 0) + 1
                for style, count in sorted(style_counts.items(),
                                          key=lambda x: x[1], reverse=True):
                    report.append(f"  - {style}: {count} ({count/len(learning_styles):.1%})")
                report.append("")
        
        # Bloom层次表现
        report.append("【Bloom认知层次整体表现】")
        bloom_names = ['记忆', '理解', '应用', '分析', '评价', '创造']
        all_final_scas = []
        
        for result in test_results:
            if result.get('final_sca'):
                all_final_scas.append(np.array(result['final_sca']))
        
        if all_final_scas:
            avg_sca = np.mean(all_final_scas, axis=0)
            overall_bloom = np.mean(avg_sca, axis=0)
            
            for i, name in enumerate(bloom_names):
                bar_length = int(overall_bloom[i] * 30)
                bar = '█' * bar_length + '░' * (30 - bar_length)
                report.append(f"{name:4s}: {bar} {overall_bloom[i]:.3f}")
        
        report.append("")
        report.append("="*70)
        report.append("报告结束")
        report.append("="*70)
        
        # 保存报告
        report_text = '\n'.join(report)
        report_path = os.path.join(save_dir, 'test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n报告已保存至: {report_path}")
        return report_text

def main():
    """主函数"""
    # 配置
    config = {
        'data_dir': '../data',
        'llm_mode': 'local',  # 'local' 或 'api'
        'local_model_path': '../LLM/qwen3/',
        'api_key': "sk-e72469ad769b4879b527f39732b6a13c",  # API模式需要
        'sca_path': None,  # 使用默认路径
        'oca_path': None,  # 使用默认路径
        'max_steps': 20,
        'convergence_threshold': 0.02
    }
    
    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.abspath(f"../results/bloomagent_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"结果将保存至: {results_dir}")
    
    # 创建CAT系统
    cat_system = BloomAgentCAT(config)
    cat_system._init_agents(results_dir)
    
    # 选择测试模式
    print("\n选择测试模式:")
    print("1. 单个学生测试")
    print("2. 批量测试")
    mode = input("请输入选择 (1/2): ").strip()
    
    if mode == '1':
        # 单个学生测试
        student_id = int(input("请输入学生ID: ").strip())
        max_steps = int(input(f"最大题目数 (默认{config['max_steps']}): ").strip() or config['max_steps'])
        
        result = cat_system.run_adaptive_test(
            student_id=student_id,
            max_steps=max_steps
        )
        
        if result:
            # 保存结果
            result_path = os.path.join(results_dir, f"student_{student_id}_result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n测试结果已保存至: {result_path}")
            
            # 生成报告
            report = cat_system.generate_report([result], results_dir)
            print("\n" + report)
    
    elif mode == '2':
        # 批量测试
        limit = input("测试学生数量 (留空测试全部): ").strip()
        limit = int(limit) if limit else None
        max_steps = int(input(f"每人最大题目数 (默认{config['max_steps']}): ").strip() or config['max_steps'])
        
        results = cat_system.batch_test(
            limit=limit,
            max_steps=max_steps
        )
        
        if results:
            # 保存所有结果
            results_path = os.path.join(results_dir, "all_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n所有结果已保存至: {results_path}")
            
            # 生成报告
            report = cat_system.generate_report(results, results_dir)
            print("\n" + report)
    
    else:
        print("无效选择")
    
    print(f"\n测试完成！所有文件保存在: {results_dir}")

if __name__ == "__main__":
    main()