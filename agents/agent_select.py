import json
import random
import os
from bloom_taxonomy import BloomLevel, get_next_level

class SelectAgent:
    """选题Agent：仅通过prompt格式调整避免ID前缀问题"""
    
    def __init__(self, llm=None, tokenizer=None,
                 chat_model="qwen3-max", q_text=True, reflection=True):
        super().__init__()
        
        self.llm = llm
        if self.llm is None:
            raise ValueError("必须提供LLM实例")
        
        self.QTEXT = q_text
        self.REFLECTION = reflection
        self._load_question_data()
        
        self.teacher_setting = (
            "你是一位经验丰富的教师，擅长根据学生的学习情况推荐合适的题目。"
            "请基于Bloom认知分类法和学生当前的学习状态，推荐最适合学生下一步学习的题目。"
        )
    
    def _load_question_data(self):
        """加载题目数据（保持原始格式）"""
        try:
            data_path = os.path.abspath("../data/processed")
            
            with open(os.path.join(data_path, "questions.json"), 'r', encoding='utf-8') as f:
                self.questions = json.load(f)
            
            with open(os.path.join(data_path, "k2q.json"), 'r', encoding='utf-8') as f:
                self.k2q = json.load(f)
            
            with open(os.path.join(data_path, "bloom2q.json"), 'r', encoding='utf-8') as f:
                self.bloom2q = json.load(f)
            
            with open(os.path.join(data_path, "q2text.json"), 'r', encoding='utf-8') as f:
                self.q2text = json.load(f)
            
            with open(os.path.join(data_path, "q2k.json"), 'r', encoding='utf-8') as f:
                self.q2k = json.load(f)
            
            with open(os.path.join(data_path, "q2bloom.json"), 'r', encoding='utf-8') as f:
                self.q2bloom = json.load(f)
            
            print(f"选题Agent成功加载 {len(self.questions)} 道题目")
            
        except Exception as e:
            print(f"选题Agent加载题目数据失败: {str(e)}")
            self.questions = {}
            self.k2q = {}
            self.bloom2q = {}
            self.q2text = {}
            self.q2k = {}
            self.q2bloom = {}
    
    def get_question_info(self, question_id):
        """获取题目详细信息（仅做基础格式处理）"""
        qid_str = str(question_id).strip()  # 仅去除空格，不做复杂替换
        if qid_str in self.questions:
            return self.questions[qid_str]
        else:
            return {
                'id': qid_str,
                'text': self.q2text.get(qid_str, f"题目 {qid_str}"),
                'knowledge_points': self.q2k.get(qid_str, []),
                'bloom_level': self.q2bloom.get(qid_str, "")
            }
    
    def get_candidate_questions(self, weaknesses, history_qids, limit=20):
        """获取候选题目（保持原始ID格式）"""
        history_qids_str = [str(qid).strip() for qid in history_qids]
        candidates = set()
        
        for weakness in weaknesses:
            kp = weakness['knowledge_point']
            bl = weakness['bloom_level']
            
            if kp in self.k2q:
                for qid in self.k2q[kp][:limit//3]:
                    qid_str = str(qid).strip()
                    if qid_str not in history_qids_str:
                        candidates.add(qid_str)
            
            bl_str = bl.value
            if bl_str in self.bloom2q:
                for qid in self.bloom2q[bl_str][:limit//3]:
                    qid_str = str(qid).strip()
                    if qid_str not in history_qids_str:
                        candidates.add(qid_str)
            
            next_level = get_next_level(bl)
            next_level_str = next_level.value
            if next_level_str in self.bloom2q:
                for qid in self.bloom2q[next_level_str][:limit//4]:
                    qid_str = str(qid).strip()
                    if qid_str not in history_qids_str:
                        candidates.add(qid_str)
        
        if len(candidates) < limit:
            all_qids = [str(qid).strip() for qid in self.questions.keys()]
            random.shuffle(all_qids)
            for qid in all_qids:
                if qid not in candidates and qid not in history_qids_str:
                    candidates.add(qid)
                    if len(candidates) >= limit:
                        break
        
        return list(candidates)[:limit]
    
    def given_advise(self, history_text, history_qids, student_profile, diagnosis, recommend_reflection=""):
        weaknesses = diagnosis.get('weaknesses', [])
        history_qids_str = [str(qid).strip() for qid in history_qids]
        candidate_qids = self.get_candidate_questions(weaknesses, history_qids_str)
        candidates_info = [self.get_question_info(qid) for qid in candidate_qids]
        
        prompt = f"{self.teacher_setting}\n\n"
        
        prompt += "学生当前的学习情况如下：\n"
        prompt += f"1. 学习能力：{student_profile.get('learning_ability', '未评估')}\n"
        prompt += f"2. 学习偏好：{student_profile.get('learning_preference', '未评估')}\n"
        prompt += "3. 薄弱环节：\n"
        for i, w in enumerate(weaknesses[:3]):
            prompt += f"   {i+1}. 知识点：{w['knowledge_point']}，认知层次：{w['bloom_level'].value}，掌握程度：{w['mastery']:.2f}\n"
        
        prompt += "\n学生的答题历史（已做题目，不可再推荐）：\n"
        for i, (q, a) in enumerate(history_text[-3:]):
            prompt += f"   {i+1}. {q}，{a}\n"
        
        prompt += f"\n已做题目ID列表（必须排除）：{history_qids_str}\n"
        
        if recommend_reflection:
            prompt += f"\n之前的推荐反思：{recommend_reflection}\n"
        
        # 关键修改1：候选题目直接以原始ID开头，不添加序号
        prompt += "\n以下是可供选择的新题目（格式：[原始ID] 题目内容）：\n"
        for q_info in candidates_info:
            kps = ", ".join(q_info.get('knowledge_points', []))
            # 直接用原始ID开头，强调ID的格式
            prompt += f"[{q_info['id']}] {q_info['text']}（知识点：{kps}，认知层次：{q_info['bloom_level']}）\n"
        
        # 关键修改2：明确要求返回原始ID，不得添加任何前缀
        prompt += "\n请从上述题目中选择最适合的1道题，并给出推荐理由，以及根据学生当前画像，预测学生对这道题的作答是否正确。返回格式必须为："
        prompt += "{'question_id': '原始ID', 'recommand_reason': '推荐理由', 'predict_answer': '正确/错误'}"
        
        advise = self.llm(prompt)
        return advise, prompt
    