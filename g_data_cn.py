from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
import DeepseekTokenizer
import torch
import os
import re
# import regex as re
import unicodedata

access_token = ''

tokenizer = DeepseekTokenizer.DeepseekTokenizer()
# tokenizer = AutoTokenizer.from_pretrained("Atom-7B_tokenizer", use_fast=True)

data_path = 'fjcanyue/wikipedia-zh-cn'
stream_dataset = load_dataset(data_path, split='train',streaming=True)

def is_mostly_chinese(text):
    """
    检查字符串中中文字符是否超过总字符数的一半。
    
    Args:
        text (str): 要检查的字符串。
        
    Returns:
        bool: 如果中文字符数量大于总字符数的一半，返回 True；否则返回 False。
    """
    # 匹配所有中文字符
    chinese_chars = re.findall("[\u4e00-\u9fa5]", text)
    
    # 过滤掉空格和控制字符，只计算可见字符
    # 这里我们简化处理，直接用原始文本长度进行比较
    if len(text) == 0:
        return False
        
    if len(chinese_chars) > len(text) / 2:
        return True
    else:
        return False
    
def punctuations_to_fullwidth(text):
    """
    智能地将半角标点符号转换为全角。

    转换逻辑：
    1. 检查标点符号的前后是否为中文。这里的“中文”包括中文字符和中文标点。
    2. 如果标点符号的前后都是数字或英文字母，则保持半角。
    3. 特殊处理引号和句号。

    Args:
        text (str): 待转换的字符串。

    Returns:
        str: 转换后的字符串。
    """
    result = list(text)

    # 定义中文标点和中文字符的匹配模式
    chinese_char_and_punctuation_pattern = r'[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]'
    
    # 匹配所有半角标点符号
    halfwidth_punctuation_pattern = r'[!\"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]'

    for i in range(len(result)):
        char = result[i]

        # 检查当前字符是否为半角标点符号
        if re.fullmatch(halfwidth_punctuation_pattern, char):
            # 获取前后字符，并处理边界情况
            prev_char = text[i-1] if i > 0 else ''
            next_char = text[i+1] if i < len(text) - 1 else ''

            # 判断前后是否都是数字或字母（或者边界）
            is_surrounded_by_alpha_or_digit = (
                (prev_char.isalnum() or not prev_char) and 
                (next_char.isalnum() or not next_char)
            )

            # 判断前后是否有中文（中文字符或中文标点）
            has_chinese_context = (
                re.search(chinese_char_and_punctuation_pattern, prev_char) or
                re.search(chinese_char_and_punctuation_pattern, next_char)
            )

            if has_chinese_context and not is_surrounded_by_alpha_or_digit:
                # 转换引号
                if char == "'":
                    result[i] = '‘' if next_char in [' ', ''] else '’'
                elif char == '"':
                    result[i] = '“' if next_char in [' ', ''] else '”'
                # 转换句号
                elif char == '.':
                    result[i] = '。'
                # 转换其他标点
                elif 33 <= ord(char) <= 126:
                    result[i] = chr(ord(char) + 65248)

    return "".join(result)


def is_reserved_or_symbol_char(c):
    cat = unicodedata.category(c)
    code = ord(c)

    if c in ['\n', '\t']:
        return False
    # 过滤控制字符、私有保留区、非法符号
    if cat.startswith('C'):  # Cc, Cf, Cs, Co, Cn
        return True
    if cat == 'So':  # Symbol, Other - 包括 █▇▆▅▄
        return True
    if 0xFFF0 <= code <= 0xFFFF:  # 私有区
        return True
    # 不可打印字符
    if not c.isprintable() and not unicodedata.category(c).startswith("Z"):
        return True
    return False

def truncate_repeats(text, max_repeats=5):
    # 匹配连续重复字符，并截断为最多 N 次
    return re.sub(r'(.)\1{' + str(max_repeats) + r',}', lambda m: m.group(1) * max_repeats, text)

def clean_line(line):
    line = line.replace('\r\n', '\n').replace('\r', '')
    # line = line.replace('\t', ' ')
    line = ''.join(c for c in line if not is_reserved_or_symbol_char(c))
    return truncate_repeats(line, max_repeats=4)

def has_special_char(text):
    for c in text:
        if is_reserved_or_symbol_char(c):
            return True
    return False

def chunk_text(input_ids, offset_mapping, chunk_size=1024):
    final_chunks = []
    current_pos = 0

    while current_pos < len(input_ids):
        # 确定初步的切分点
        end_pos = min(current_pos + chunk_size, len(input_ids))

        # 如果切分点不是文本末尾，就需要检查并调整边界
        if end_pos < len(input_ids):
            # 从切分点回退，直到找到一个有效的字符边界
            # 有效边界：切分点 token 和前一个 token 不属于同一个原始字符
            while end_pos > current_pos and offset_mapping[end_pos][0] == offset_mapping[end_pos - 1][0]:
                end_pos -= 1

        # --- 这里是关键的修正 ---
        # 检查是否发生了死循环的情况 (end_pos 被推回了起点)
        if end_pos == current_pos:
            # 这种情况发生于：单个字符的 token 数量 > chunk_size，
            # 或者像你的例子中，一个多 token 字符位于 chunk 的最开始。
            # 我们必须强制前进，将这整个多 token 的字符作为一个 chunk。
            
            # 获取当前字符的 offset start
            start_char_offset = offset_mapping[current_pos][0]
            
            # 向前找到这个字符的结束位置
            end_pos = current_pos + 1
            while end_pos < len(input_ids) and offset_mapping[end_pos][0] == start_char_offset:
                end_pos += 1
                
            # 如果这个超长字符直接到了文本末尾，确保我们不会超出范围
            if end_pos >= len(input_ids):
                end_pos = len(input_ids)

        # 提取最终的 chunk
        chunk = input_ids[current_pos:end_pos]
        
        # 确保 chunk 不为空再添加
        if chunk:
            final_chunks.append(torch.tensor(chunk))

        # 更新下一个 chunk 的起始位置
        current_pos = end_pos
    return final_chunks

def remove_empty_brackets(text):
    """
    迭代地删除所有空的或只包含空白的括号对。
    例如: "你好 () 世界" -> "你好 世界"
           "【（ ）】" -> ""
    """
    # 定义需要处理的中英文括号对
    bracket_pairs = [
        ("(", ")"), ("（", "）"),
        ("[", "]"), ("【", "】"),
        ("{", "}"), ("｛", "｝"),
        ("「", "」"), ("『", "』")
    ]
    
    # 使用循环，直到文本不再发生变化
    while True:
        # 记录处理前的文本长度，用于判断是否需要继续循环
        len_before = len(text)
        
        for open_b, close_b in bracket_pairs:
            # 创建正则表达式，匹配 "开括号 + 0个或多个空白 + 闭括号"
            # 使用 re.escape 来处理括号这种在正则中有特殊含义的字符
            pattern = rf"{re.escape(open_b)}\s*{re.escape(close_b)}"
            text = re.sub(pattern, "", text)
            
        # 如果处理后文本长度没有变化，说明已经清理干净，跳出循环
        if len(text) == len_before:
            break
            
    return text.strip() # 最后再去除一下可能产生的首尾空格

def clean_text_for_llm_ultimate(text):
    """
    一个终极版的LLM预训练文本清洗函数，包含短语去重。
    """
    if not isinstance(text, str) or not text:
        return ""

    # 1. 统一换行符
    text = text.replace('\r\n', '\n')

    # 2. 去除4次以上的重复字符/标点/换行
    text = re.sub(r'([^\n])\1{3,}', r'\1\1\1', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # 3. 【新增】去除重复的词语/短语 (例如 "你好你好" -> "你好")
    # 这个模式会匹配任何连续重复两次或以上的、长度至少为2的字符串
    # text = re.sub(r'(.{2,})\1+', r'\1', text)
    # text = re.sub(r'(.{2,})(?>\1+)', r'\1', text)

    # 4. 【新增】间隔性短语去重
    # 定义分隔符
    # separators = r'\s*[！，。？?,!.;]\s*'
    # 匹配 (词组) + (分隔符 + 词组) 的重复模式
    # 例如：匹配 "加油！加油" 或 "你好， 你好"
    # text = re.sub(rf'(.{{2,}})({separators}\1)+', r'\1', text)
    # text = re.sub(rf'(.{{2,}})(?>(?:{separators}\1)+)', r'\1', text)
    
    # 4. Unicode 归一化 (NFC)
    text = unicodedata.normalize('NFC', text)

    # 5. 转换全角数字和字母为半角
    full_to_half_table = str.maketrans(
        "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    )
    text = text.translate(full_to_half_table)

    # 6. 字符白名单过滤
    emoji_pattern = (
        r'\U0001F600-\U0001F64F' r'\U0001F300-\U0001F5FF' r'\U0001F680-\U0001F6FF' r'\U0001F700-\U0001F77F'
        r'\U0001F780-\U0001F7FF' r'\U0001F800-\U0001F8FF' r'\U0001F900-\U0001F9FF' r'\U0001FA70-\U0001FAFF'
        r'\u2600-\u26FF' r'\u2700-\u27BF' r'\uFE0F'
    )
    keep_chars_pattern = re.compile(
        rf'[^\u4e00-\u9fa5a-zA-Z0-9\n'
        rf'\u3000-\u303F\uFF00-\uFFEF' # 中文标点
        rf'.,!?;:\'"()\[\]{{}}<>\+\-\*/=「」『』' # 常用标点
        rf'\u2460-\u24FF'  # <--- 新增：带圈字母和数字 (Enclosed Alphanumerics)
        rf'{emoji_pattern}]'
    )
    text = keep_chars_pattern.sub('', text)
    
    # 7. 结构级清洗：移除空括号
    text = remove_empty_brackets(text)

    # 8. 标准化水平空白符
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


current_input = []
current_target = []
g1024 = 0
g512 = 0
g256 = 0
g128 = 0
g0 = 0
less_10 = 0
total_tokens = 0
tokens = 0
token_total = 0
special_count = 0
unknown_count = 0
def stream_and_chunk(dataset, chunk_size=512, stride=128):
    global g1024
    global g512
    global g256
    global g128
    global g0
    global total_tokens
    global tokens
    global token_total
    global less_10
    global special_count
    global unknown_count
    for e in dataset:
        # if has_special_char(e['text']):
        #     special_count += 1
        #     continue
        # for e in example['text']:
        i = 0
        # e['text'] = clean_line(e['text'])
        text = e['text'] + tokenizer.eos_token
        token_ids = tokenizer.encode(text)
        # encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        # token_ids = encoding["input_ids"]
        # offset_mapping = encoding["offset_mapping"]
        remaining_size = token_ids.shape[0]
        total_tokens += remaining_size
        unknown = (token_ids==tokenizer.unk_token_id).sum()
        if unknown>0:
            unknown_count +=1
            continue
        if remaining_size>=1024:
            g1024 +=1
        elif remaining_size>=512:
            g512 +=1
        elif remaining_size>=256:
            g256 +=1
        elif remaining_size>=128:
            g128 +=1
        else:
            g0 +=1
        # if remaining_size<1024:
            # continue
        tokens += remaining_size
        # final_chunks = chunk_text(token_ids, offset_mapping, chunk_size)
        # for each_chunk in final_chunks:
        # tmp = torch.full([chunk_size-token_ids.shape[0]], tokenizer.pad_token_id)
        # token_ids = torch.cat((token_ids, tmp))
        # yield {"input":token_ids}
        while remaining_size>0:
            if (remaining_size<=chunk_size):
                input_chunk = token_ids[i:]
                # target_chunk = token_ids[i + 1: ]
                tmp = torch.full([chunk_size-input_chunk.shape[0]+1], tokenizer.pad_token_id)
                input_chunk = torch.cat((input_chunk, tmp))
                token_total += (input_chunk != tokenizer.pad_token_id).sum().item()
                # tmp = torch.full([chunk_size-target_chunk.shape[0]], tokenizer.pad_token_id)
                # target_chunk = torch.cat((target_chunk, tmp))
            else:
                input_chunk = token_ids[i:i + chunk_size+1]
                token_total += input_chunk.shape[0] - 1
                # target_chunk = token_ids[i + 1: i + chunk_size + 1]
            # yield {"input":tokenizer.decode(input_chunk.detach().clone()), "target":tokenizer.decode(target_chunk.detach().clone())}
            
            # yield {"input":input_chunk, "target":target_chunk}
            # unknown = (input_chunk==tokenizer.unk_token_id).sum()
            # if remaining_size>10:
            #     yield {"input":input_chunk}
            # elif unknown>0:
            #     unknown_count += 1
            # else:
            #     less_10 += 1
            # if unknown>0:
                # unknown_count += 1
            # else:
            yield {"input":input_chunk}
            # res.append({"i":input_chunk.detach().clone(), "t":target_chunk.detach().clone()})
            i = i+stride
            remaining_size = remaining_size -stride

        # yield {"id":example['id']}

# chunked_dataset = Dataset.from_generator(stream_and_chunk, gen_kwargs={"dataset":stream_dataset, "chunk_size":1024, "stride":1024}, cache_dir='', num_proc=8)
chunked_dataset = Dataset.from_generator(stream_and_chunk, gen_kwargs={"dataset":stream_dataset, "chunk_size":1024, "stride":1024})
# chunked_dataset = chunked_dataset.shard(8)
# for i,b in enumerate(chunked_dataset):
    # aaa = 1
    # print(tokenizer.decode(b['input']))

chunked_dataset.save_to_disk("")
print(g1024, g512, g256, g128, g0, end="\n")
print(f"total token: {tokens}/{total_tokens}, {token_total}")
print(f"special_count:{special_count} unknown:{unknown_count}")