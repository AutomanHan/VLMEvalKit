from transformers import StoppingCriteria, StoppingCriteriaList

# 1. 定义自定义停止条件
class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_strings,prompt, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_length = prompt["input_ids"].shape[-1]
        self.stop_reason= None

    def __call__(self, input_ids, score, **kwargs):
        # 解码当前生成的文本
        current_text = self.tokenizer.decode(input_ids[0][self.prompt_length:], skip_special_tokens=True)
        
        # 检查是否包含任意终止字符串
        # 检查每个终止字符串
        for stop_str in self.stop_strings:
            if stop_str in current_text:
                self.stop_reason = f"stopped_by_{stop_str}"  # 记录具体终止字符串
                return True  # 触发停止
        return False  # 继续生成
    def __len__(self):
        return 1  # 必须实现的方法