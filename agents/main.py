import json
import os
import random
import re
import warnings
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
import numpy as np

from llm import QwenLLM,LocalQwenLLM
from agent_diagnose import DiagnoseAgent
from agent_select import SelectAgent

warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma`")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta`")

def run_cat(model, max_step, reflection=True, q_text=True, llm=None):
    """运行基于Bloom认知维度的双Agent CAT系统，结果存储在独立文件夹"""
    # 生成唯一时间戳（确保每次运行文件夹独立）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 主结果目录：BloomAgent/results/[时间戳]/
    main_results_dir = os.path.abspath(f"../results/{timestamp}")
    # 子目录：日志文件、学生画像、结果JSON
    log_dir = os.path.join(main_results_dir, "logs")
    memory_dir = os.path.join(main_results_dir, "memory")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(memory_dir, exist_ok=True)
    
    # 配置参数
    COLD_NUM = 5
    STUDENT_NUM = 1
    MAX_STEP = max_step
    CHAT_MODEL = model
    # 预处理数据仍从data/processed读取（输入数据不变）
    datapath = os.path.abspath("../data/processed")
    
    # 检查输入数据是否存在
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"未找到预处理数据，请先运行preprocess_data.py。路径：{datapath}")
    
    # 加载知识点和学生数据
    with open(os.path.join(datapath, "knowledge_points.json"), 'r', encoding='utf-8') as f:
        knowledge_points = json.load(f)
    
    with open(os.path.join(datapath, "students.json"), 'r', encoding='utf-8') as f:
        students_data = json.load(f)
        student_count = min(STUDENT_NUM, len(students_data.keys()))
        students = list(random.sample(students_data.keys(), student_count)) if student_count > 0 else []
    
    if not students:
        print("警告：未找到学生数据，程序将退出")
        return
    
    # 记录所有学生的结果
    all_results = {}
    # 日志文件路径：results/[时间戳]/logs/cat_results.txt
    output_file = os.path.join(log_dir, f"cat_results_{model}_{max_step}.txt")
    
    with open(output_file, "w", encoding='utf-8') as f:
        try:
            print(f"本次运行结果将保存至：{main_results_dir}\n")
            f.write(f"本次运行结果路径：{main_results_dir}\n\n")
            
            for student_id in tqdm(students, desc="处理学生"):
                # 初始化诊断Agent，指定学生画像存储路径为当前结果的memory目录
                diagnose_agent = DiagnoseAgent(
                    llm=llm, 
                    q_text=q_text,
                    reflection=reflection,
                    mem_file_path=memory_dir  # 学生画像存储到独立文件夹
                )
                diagnose_agent.set_student_id(student_id)
                
                select_agent = SelectAgent(
                    llm=llm, 
                    q_text=q_text,
                    reflection=reflection
                )
                
                # 冷启动数据
                initial_data = students_data[student_id].get('initial_data', {})
                initial_ques = initial_data.get('questions', [])[:COLD_NUM]
                initial_ans = initial_data.get('answers', [])[:COLD_NUM]
                
                print(f"\n处理学生 {student_id}，初始题目: {initial_ques[:3]}...")
                f.write(f"\n处理学生 {student_id}，初始题目: {initial_ques[:3]}...\n")
                
                for i in range(min(COLD_NUM, len(initial_ques), len(initial_ans))):
                    diagnose_agent.update_history_log(
                        initial_ques[i], 
                        initial_ans[i],
                        "冷启动初始题目",
                        ""
                    )
                
                # 自适应测试循环
                student_results = []
                for step in range(MAX_STEP):
                    print(f"\n===== 步骤 {step+1}/{MAX_STEP} =====")
                    f.write(f"\n===== 步骤 {step+1}/{MAX_STEP} =====\n")
                    
                    # 获取诊断结果
                    diagnosis = diagnose_agent.get_diagnosis()
                    student_profile = diagnose_agent.get_student_profile()
                    history_text = diagnose_agent.get_history_text()
                    # 关键修改：获取完整的历史题目ID列表（包括冷启动阶段）
                    history_qids = diagnose_agent.memory['history_log']['question_id']
                    
                    # 反思逻辑
                    if reflection:
                        diagnose_agent.reflection()
                        recommend_reflection = diagnose_agent.memory['recommend_reflection']
                        print(f"反思结果: {recommend_reflection}")
                        f.write(f"反思结果: {recommend_reflection}\n")
                    else:
                        recommend_reflection = ""
                    
                    # 将历史题目ID传入选题Agent
                    advise, prompt = select_agent.given_advise(
                        history_text,
                        history_qids,  # 传递完整的历史题目ID列表
                        student_profile,
                        diagnosis,
                        recommend_reflection
                    )
                    
                    # 解析推荐结果
                    question_id = None
                    recommand_reason = ""
                    predict_answer = ""
                    
                    try:
                        advise_json = json.loads(advise)
                        question_id = advise_json.get('question_id')
                        recommand_reason = advise_json.get('recommand_reason', "")
                        predict_answer = advise_json.get('predict_answer', "")
                    except:
                        try:
                            qid_match = re.search(r"'question_id': '?(\w+)'?", advise)
                            if qid_match:
                                question_id = qid_match.group(1)
                        except:
                            # 从候选池中随机选择一个未选过的题目
                            candidate_qids = select_agent.get_candidate_questions(
                                diagnosis.get('weaknesses', []), 
                                history_qids
                            )
                            if candidate_qids:
                                question_id = random.choice(candidate_qids)
                            else:
                                question_id = random.choice(list(select_agent.questions.keys()))
                
                    if not question_id:
                        print("无法获取有效题目ID，结束测试")
                        f.write("无法获取有效题目ID，结束测试\n")
                        break
                    
                    # 处理答题结果
                    question_id = str(question_id)
                    # 再次检查是否为已选题目（双重保险）
                    if question_id in [str(qid) for qid in history_qids]:
                        print(f"检测到重复推荐，自动替换题目 {question_id}")
                        # 自动替换为其他题目
                        candidate_qids = select_agent.get_candidate_questions(
                            diagnosis.get('weaknesses', []), 
                            history_qids
                        )
                        if candidate_qids:
                            question_id = random.choice(candidate_qids)
                        else:
                            question_id = random.choice(list(select_agent.questions.keys()))
                    
                    question_info = select_agent.get_question_info(question_id)
                    student_answers = students_data[student_id].get('answers', {})
                    answer_bi = student_answers.get(question_id, random.choice([0, 1]))
                    
                    print(f"推荐题目: {question_info['text'][:100]} (ID: {question_id})")
                    print(f"学生答案: {'正确' if answer_bi == 1 else '错误'}")
                    f.write(f"推荐题目: {question_info['text'][:100]} (ID: {question_id})\n")
                    f.write(f"学生答案: {'正确' if answer_bi == 1 else '错误'}\n")
                    
                    # 更新诊断Agent
                    diagnose_agent.update_history_log(
                        question_id,
                        answer_bi,
                        recommand_reason,
                        predict_answer
                    )
                    
                    student_results.append({
                        'step': step + 1,
                        'question_id': question_id,
                        'question_text': question_info['text'],
                        'knowledge_points': question_info['knowledge_points'],
                        'bloom_level': question_info['bloom_level'],
                        'answer': answer_bi,
                        'predict_answer': predict_answer
                    })
                
                # 保存学生结果
                all_results[student_id] = student_results
                
                # 保存学生画像（路径已指向当前结果的memory目录）
                profile_path = diagnose_agent.save_memory()
                print(f"学生 {student_id} 的画像已保存至: {profile_path}")
                f.write(f"学生 {student_id} 的画像已保存至: {profile_path}\n")
            
            # 保存所有结果到JSON：results/[时间戳]/all_cat_results.json
            results_json_path = os.path.join(main_results_dir, "all_cat_results.json")
            with open(results_json_path, 'w', encoding='utf-8') as res_file:
                json.dump(all_results, res_file, ensure_ascii=False, indent=4)
            print(f"\n所有测试结果已保存至: {results_json_path}")
            f.write(f"\n所有测试结果已保存至: {results_json_path}\n")
        
        except Exception as e:
            error_msg = f"运行过程中出错: {str(e)}"
            print(error_msg)
            f.write(error_msg + "\n")
        finally:
            print(f"CAT测试完成！所有结果位于: {main_results_dir}")
            f.write(f"CAT测试完成！所有结果位于: {main_results_dir}\n")
        
        return all_results

if __name__ == "__main__":
    parser = ArgumentParser(description="基于Bloom认知维度的双Agent CAT系统")
    parser.add_argument("--max_step", type=int, default=10, help="最大测试步骤")
    parser.add_argument("--no-reflection", action="store_false", dest="reflection", help="禁用反思功能")
    parser.add_argument("--no-qtext", action="store_false", dest="q_text", help="不显示题目文本")
    
    args = parser.parse_args()
    
    # ==================== 切换LLM方式：修改这里 ====================
    # 方式1：使用API（默认）
    llm = QwenLLM(
       model_name="qwen3-max",
        api_key="sk-e72469ad769b4879b527f39732b6a13c",
        temperature=0.3,          # 降低随机性
        max_tokens=8192,          # 最大输出长度
        enable_thinking=True,     # 开启深度思考
    )
    model_name = "qwen3-max"
    
    # 方式2：使用本地模型（取消下面的注释，注释掉上面的代码）
    # llm = LocalQwenLLM(
    #     model_path="../LLM/Qwen3-8B/",temperature=0.3,max_tokens=8192
    # )
    # model_name = "local-Qwen3-8B"
    # ============================================================
    
    # 运行系统
    run_cat(
        model=model_name,
        max_step=args.max_step,
        reflection=args.reflection,
        q_text=args.q_text,
        llm=llm
    )