from enum import Enum

class BloomLevel(Enum):
    """Bloom认知目标分类法的六个层次"""
    REMEMBER = "记忆"         # 记忆：回忆或识别信息
    UNDERSTAND = "理解"       # 理解：解释或总结信息
    APPLY = "应用"            # 应用：在具体情境中使用信息
    ANALYZE = "分析"          # 分析：将信息分解为部分
    EVALUATE = "评价"         # 评价：基于标准进行判断
    CREATE = "创造"           # 创造：将元素整合为新的形式

    @classmethod
    def from_string(cls, level_str):
        """从字符串转换为BloomLevel枚举"""
        if not level_str:
            return None
            
        # 直接匹配
        for level in cls:
            if level.value == level_str:
                return level
        
        # 模糊匹配（处理同义词）
        synonyms = {
            "记忆": ["记住", "回忆", "识别"],
            "理解": ["领会", "解释", "总结"],
            "应用": ["使用", "运用", "应用"],
            "分析": ["分解", "分析", "区分"],
            "评价": ["判断", "评估", "评价"],
            "创造": ["创作", "构建", "创造"]
        }
        
        for level in cls:
            if level_str in synonyms[level.value]:
                return level
                
        return None
    
    @classmethod
    def from_number(cls, level_num):
        """从数字(1-6)转换为BloomLevel枚举"""
        mapping = {
            1: cls.REMEMBER,
            2: cls.UNDERSTAND,
            3: cls.APPLY,
            4: cls.ANALYZE,
            5: cls.EVALUATE,
            6: cls.CREATE
        }
        return mapping.get(level_num, None)
    
    def to_number(self):
        """将枚举转换为对应的数字(1-6)"""
        mapping = {
            self.REMEMBER: 1,
            self.UNDERSTAND: 2,
            self.APPLY: 3,
            self.ANALYZE: 4,
            self.EVALUATE: 5,
            self.CREATE: 6
        }
        return mapping.get(self, 0)

def get_bloom_hierarchy(level):
    """获取某个认知层次在Bloom分类中的层级(1-6)"""
    if isinstance(level, BloomLevel):
        return level.to_number()
    return 0

def level_difference(level1, level2):
    """计算两个认知层次之间的差异"""
    return abs(get_bloom_hierarchy(level1) - get_bloom_hierarchy(level2))

def get_next_level(level):
    """获取当前层次的下一个更高层次"""
    current_num = get_bloom_hierarchy(level)
    if current_num < 6:
        return BloomLevel.from_number(current_num + 1)
    return level  # 最高层次返回自身

def get_previous_level(level):
    """获取当前层次的上一个更低层次"""
    current_num = get_bloom_hierarchy(level)
    if current_num > 1:
        return BloomLevel.from_number(current_num - 1)
    return level  # 最低层次返回自身
    