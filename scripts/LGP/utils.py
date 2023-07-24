from re import A
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, DebertaV2Tokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import sys
import json
import random
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# --- 5

tokenizer = RobertaTokenizer.from_pretrained("/PretrainedModels/roberta-large")
tokenizer.add_special_tokens({'additional_special_tokens':["[LGMASK]"]})


premise_indicators = ["given that", "seeing that", "for the reason that", "owing to", "as indicated by", "after all", "on the grounds that", "since", "on account of", "considering", "because of","because", "due to", "now that", "as indicated by", "may be inferred from", "by virtue of", "in view of", "for the sake of", "thanks to", "thanked to", "for this reason that", "as long as", "based on that", "this is because", "this was because", "as a result of", "considering that", "inasmuch as", "if and only if", "according to", "in that", "only if", "once", "depend on", "rely on"]


conclusion_indicators = ["conclude that", "concluded that", "concluding that", "it entails", "we may infer that", "it implies that", "that is why", "therefore", "thereby", "wherefore", "accordingly", "entails that", "hence", "thus", "consequently", "we may infer", "it must be that", "whence", "so that", "so", "it follows that", "implies that", "as a result", "it can be inferred that", "suggests that", "proves that", "it can be shown", "as a conclusion", "conclusively", "which implies that", "for that reason", "as a consequence", "on that account", "that being said", "in conclusion", "to that end", "for this reason", "because of this", "that being so", "because of that", "ergo", "in this way", "in this manner", "in such a manner", "by such means", "as it turns out", "result in", "resulted in", "resulting in", "in order", "in order to", "in order that", "show that", "eventually"]

neg_indicators = ["not", "neither", "none of", "unable", "few", "little", "hardly", "merely", "seldom", "without", "never", "nobody", "nothing", "nowhere", "rarely", "scarcely", "barely", "no longer", "isn't", "aren't", "wasn't", "weren't", "can't", "cannot", "couldn't", "won't", "wouldn't", "don't", "doesn't", "didn't", "haven't", "hasn't"]

turn_indicators = ["although", "though", "but", "nevertheless", "however", "instead", "instead of", "nonetheless", "yet", "rather", "whereas", "otherwise", "conversely", "on the contrary", "even", "nevertheless", "despite", "in spite of", "in contrast", "even if", "even though", "unless", "regardless of", "reckless of"]

and_indicators = ["and", "or", "nor", "also", "moreover", "in addition", "on the other hand", "meanwhile", "further", "afterward", "afterwards", "next", "besides", "additionally", "meantime", "furthermore", "as well", "simultaneously", "either", "both", "similarly", "likewise"]


whole_indicators = premise_indicators + conclusion_indicators + neg_indicators + and_indicators + turn_indicators
indicators = whole_indicators

indicator_types = ["premise", "conclusion", "neg", "turn", "and"]
separator_within_sent = [',', ';']


# def whether_in_neg_indicator(subtoken):
#     for indicator in neg_indicators:
#         indicator_tokens = [x.replace("Ġ","") for x in tokenizer.tokenize(indicator)]
#         for token in indicator_tokens:
#             if subtoken == token:
#                 return True
#     return False
# print(whether_in_neg_indicator())


def check_contain_indicator(sentence):

    sentence = sentence.lower()

    for indicator in indicators:
        if (indicator in sentence and len(indicator.split(' ')) > 1) or (f" {indicator} " in sentence) or (sentence.startswith(f"{indicator} ")):   
            # print(indicator)
            return True
    return False

def calculate_logic_indicators_num(sentence):

    indicator_num = 0
    sentence = sentence.lower()

    for indicator in indicators:
        while (indicator in sentence and len(indicator.split(' ')) > 1) or (f" {indicator} " in sentence) or (sentence.startswith(f"{indicator} ")):
            indicator_num += 1
            sentence = sentence.replace(indicator, '', 1)

    return indicator_num

def token_num_statistics(lens, save_name):

    lens.sort()
    ratios = [x*1.0/len(lens) for x in range(1, len(lens)+1)]

    plt.figure(figsize=(18,6))
    plt.plot(lens, ratios)
    plt.xlabel("Num of Tokens")
    plt.ylabel("Accumulated Probability")
    plt.savefig(save_name)
    # plt.show()

def removeGfromList(lis):
    return_list = [x.replace("Ġ", "") if x != "Ġ" else x for x in lis]
    while "Ġ" in return_list:
        return_list.remove("Ġ")
    return return_list

def remove_fromList(lis):
    return_list = [x.replace("▁", "") if x != "▁" else x for x in lis]
    while "▁" in return_list:
        return_list.remove("▁")
    return return_list

# 定位句中逻辑关键词的位置  [左闭右闭]
def locate_indicators_in_sentence(sentence, sentence_tokens, indicators, indicators_tokens):
    
    sentence = sentence.lower()
    sentence_tokens = [x.lower() for x in sentence_tokens]

    locations = []
    
    if isinstance(tokenizer, RobertaTokenizer):
        # print("使用RobertaTokenizer")
        new_indicators = indicators + [f" {x}" for x in indicators]
        new_indicators_tokens = indicators_tokens + [removeGfromList(y) for y in [tokenizer.tokenize(f" {x}") for x in indicators]]
    elif isinstance(tokenizer, DebertaV2Tokenizer):
        new_indicators = indicators + [f" {x}" for x in indicators]
        new_indicators_tokens = indicators_tokens + [remove_fromList(y) for y in [tokenizer.tokenize(f" {x}") for x in indicators]]
    else:
        new_indicators = indicators
        new_indicators_tokens = indicators_tokens

    for i in range(len(new_indicators)):

        indicator = new_indicators[i]

        if indicator in sentence:
            indicator_tokens = new_indicators_tokens[i]
            indicator_token_num = len(indicator_tokens)
            start_idx = 0
            
            while start_idx <= len(sentence_tokens)-indicator_token_num:
                if sentence_tokens[start_idx: start_idx+indicator_token_num] == indicator_tokens:
                    locations.append((start_idx, start_idx+indicator_token_num-1))
                    start_idx = start_idx+indicator_token_num
                else:
                    start_idx += 1

    locations = sorted(locations)


    final_locations = []
    for location in locations:
        if final_locations and final_locations[-1][0] == location[0]:
            final_locations[-1] = location
        else:
            final_locations.append(location)

    # 反着过一遍
    real_final_locations = []
    for i in range(len(final_locations)-1, -1, -1):
        if real_final_locations and real_final_locations[-1][1] == final_locations[i][1]:
            real_final_locations[-1] = final_locations[i]
        else:
            real_final_locations.append(final_locations[i])
    
    # 把连着的和overlap的tuple合并
    real_final_locations = sorted(real_final_locations)

    return_locations = []
    idx = 0

    while idx < len(real_final_locations):
        if not return_locations:
            return_locations.append(real_final_locations[idx])
        elif (real_final_locations[idx][0] > return_locations[-1][0] and real_final_locations[idx][0] <= return_locations[-1][1]): #[(1,3), (2,4)]  # overlap的indicator认为是同一类型, 则取最大range
            return_locations[-1] = (return_locations[-1][0], max(return_locations[-1][1], real_final_locations[idx][1]))
        else:
            pass 
        idx += 1
    
    return return_locations if return_locations else real_final_locations



def check_indicator_polarity(indicator):

    indicator = indicator.replace("Ġ", "").strip().lower() 

    if indicator in premise_indicators:
        return 0
    elif indicator in conclusion_indicators:
        return 1
    else:
        # 是组合式indicator, 需要反向判断
        is_premise, is_conclusion = False, False
        for premise_indicator in premise_indicators:
            if premise_indicator in indicator:
                is_premise = True
        for conclusion_indicator in conclusion_indicators:
            if conclusion_indicator in indicator:
                is_conclusion = True
        
        if is_premise == is_conclusion:
            raise TypeError(f"当前indicator是{indicator}, 极性判断不清！")
        else:
            if is_premise:
                return 0
            if is_conclusion:
                return 1
            
# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

wnl = WordNetLemmatizer()

def lemmatize(sentence):
    tokens = word_tokenize(sentence)  # 分词
    tagged_sent = pos_tag(tokens)     # 获取单词词性

    lemmas_sent = []
    for i in range(len(tagged_sent)):
        tag = tagged_sent[i]
        wordnet_pos = get_wordnet_pos(tag[1])
        if wordnet_pos == wordnet.VERB:
            # 只还原动词
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
        else:
            lemmas_sent.append(tokens[i])

    return ' '.join(lemmas_sent)



def take_first_idx(x):
    return x[0][0]
def take_last_idx(x):
    return x[0][1]

def keep_max_span_for_each_start_idx(locations):
    if len(locations) == 1: return locations

    result_locations = []
    repeated_first_idx, repeated_first_idx_locations_list = -1, []

    for i in range(len(locations)-1):
        cur_location, next_location = locations[i], locations[i+1]
        cur_start_idx, cur_end_idx = cur_location[0]
        next_start_idx, next_end_idx = next_location[0]

        if cur_start_idx < next_start_idx and cur_start_idx != repeated_first_idx:
            result_locations.append(locations[i])
        elif cur_start_idx == repeated_first_idx:
            repeated_first_idx_locations_list[-1].append(cur_location)
        elif cur_start_idx == next_start_idx:
            repeated_first_idx_locations_list.append([cur_location])
            repeated_first_idx = cur_start_idx
        else:
            print(cur_location, next_location, repeated_first_idx, repeated_first_idx_locations_list)
            sys.exit()
    
    for repeated_first_idx_locations in repeated_first_idx_locations_list:
        repeated_first_idx_locations.sort(key=take_last_idx)
        result_locations.append(repeated_first_idx_locations.pop())
    result_locations.sort(key=take_first_idx)

    res_final_start_idx, res_final_end_idx = result_locations[-1][0]
    ipt_final_start_idx, ipt_final_end_idx = locations[-1][0]
    if ipt_final_start_idx > res_final_start_idx:
        result_locations.append(locations[-1])
    else:
        # 留最大span
        if ipt_final_end_idx > res_final_end_idx:
            result_locations.pop()
            result_locations.append(locations[-1])
    
    return result_locations


def no_lgmask_around(tokens, idx):
    for i in range(1, 5):
        if ((idx-i) >= 0 and ('[LGMASK]' in tokens[idx-i])) or ((idx+i) <= len(tokens)-1 and ('[LGMASK]' in tokens[idx+i])):
            return False
    return True

# 找出当前sent中包含的某类指示词
def find_indicator(sent, indicator_type):
    if indicator_type not in indicator_types:
        raise TypeError(f"传入indicator_type有误：{indicator_type}")

    indicators = globals()[f"{indicator_type}_indicators"]
    for indicator in indicators:
        if indicator in sent:
            return indicator
    return None

def count_indicator_num(sent, indicator_type):
    cur_indicator = find_indicator(sent, indicator_type)
    if not cur_indicator: return 0

    count = 0
    while cur_indicator:
        count += 1
        sent = sent.replace(cur_indicator, "")
        cur_indicator = find_indicator(sent, indicator_type)
    return count



def extract_pages(ipt_dir, opt_dir, tokenizer):

    lens, final_pages = []
    save_idx = 0

    for folder in tqdm(glob(f"{ipt_dir}/*")):
        
        for file in glob(f"{folder}/*"):
            with open(file, 'r') as f:
                for line in f.readlines():
                    data = json.loads(line)
                    text_candidate = data["text"].strip()
                    if len(text_candidate.split()) >= 20:

                        final_pages.append(json.dumps({"page":text_candidate}))
                        lens.extend([len(tokenizer.tokenize(paragraph)) for paragraph in text_candidate.split('\n')])

                        if len(final_pages) == 20000:
                            with open(f"{opt_dir}/wiki-pages-{save_idx}.jsonl", 'w') as opt_f:
                                opt_f.write('\n'.join(final_pages))
  
                            save_idx += 1
                            final_pages = []
    
    if len(final_pages) != 0:
        with open(f"{opt_dir}/wiki-pages-{save_idx}.jsonl", 'w') as opt_f:
            opt_f.write('\n'.join(final_pages))
  



def extract_para_from_page(ipt_dir, opt_dir):

    save_idx, paragraphs = 0, []
    paragraph_num, paragraph_with_logic_num = 0, 0

    for file_path in tqdm(glob(f"{ipt_dir}/wiki-pages-*.jsonl")):
        with open(file_path, 'r') as f:
            for line in f.readlines():
                page = json.loads(line)['page']
                paragraphs_in_page = page.strip().split("\n")
                paragraphs_in_page = [x for x in paragraphs_in_page if len(x.split()) > 5]
                paragraphs.extend(paragraphs_in_page)

                for para in paragraphs_in_page:
                    if check_contain_indicator(para):
                        paragraph_with_logic_num += 1
                    paragraph_num += 1

                if len(paragraphs) > 1000000:
                    
                    with open(f"{opt_dir}/wiki-paragraphs-{save_idx}.txt", 'w') as opt_f:
                        opt_f.write('\n'.join(paragraphs[:10000000]))

                    save_idx += 1
                    paragraphs = paragraphs[10000000:]

    with open(f"{opt_dir}/wiki-paragraphs-{save_idx}.txt", 'w') as opt_f:
        opt_f.write('\n'.join(paragraphs[:10000000]))

def extract_para_with_logic(ipt_dir, opt_dir):

    save_idx, logic_paragraphs, logic_paragraphs_lens, logic_indicator_appear_num = 0, [], [], [0 for _ in range(500)]
    logic_para_num = 0

    for file_path in tqdm(glob(f"{ipt_dir}/wiki-paragraphs-*.txt")):
        with open(file_path, 'r') as f:
            for line in f.readlines():

                paragraph = line.strip()

                if check_contain_indicator(paragraph):

                    logic_para_num += 1    
                    logic_paragraphs.append(paragraph)
                    logic_paragraphs_lens.append(len(tokenizer.tokenize(paragraph)))
                    logic_indicator_appear_num[calculate_logic_indicators_num(paragraph)] += 1

                    if len(logic_paragraphs) > 1000000:
                        
                        with open(f"{opt_dir}/wiki-logical-paragraphs-{save_idx}.txt", 'w') as opt_f:
                            opt_f.write('\n'.join(logic_paragraphs[:1000000]))

                        save_idx += 1
                        logic_paragraphs = logic_paragraphs[1000000:]

    if len(logic_paragraphs) != 0:
        with open(f"{opt_dir}/wiki-ogical-paragraphs-{save_idx}.txt", 'w') as opt_f:
            opt_f.write('\n'.join(logic_paragraphs[:1000000]))


# --- no 3.5 -- now 4

LOGIC_MASKED_RATIO = 0.7
NOT_LOGIC_INDICATOR_MASKED_RATIO = 0.0055 * 0.6
lcp_label_mapping = {"premise":0, "conclusion":1, "neg":2, "turn":3, "and":4, "no":5}


def mask_tagged_paragraphs(ipt_dir, opt_dir):


    label_base = list(lcp_label_mapping.values())
    lcp_label_num = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}


    indicators_tokens = {"premise":[], "conclusion":[], "neg":[], "turn":[], "and":[]}
    for key in indicators_tokens.keys():
        for indicator in locals()[f"{key}_indicators"]:
            indicators_tokens[key].append(remove_fromList(tokenizer.tokenize(indicator)))

    samples = []
    abandoned_sample_num = 0
    valid_logic_sample_num = 0

    is_abandon = False

    for file_path in glob(f"{ipt_dir}/*.jsonl"):
        save_idx = file_path.split('-')[-1].split('.')[0]

        with open(file_path, 'r') as ipt_f:
            
            sample_idx = 0

            for line in tqdm(ipt_f.readlines()):
                sample_idx += 1

                sample = json.loads(line)

                paragraph = sample['text']
                paragraph_tokens = tokenizer.tokenize(paragraph)

                paragraph_tokens_noG = [x.replace("Ġ", "").lower() if x != 'Ġ' else x for x in paragraph_tokens]

                logic_polarity_label = [-100 for _ in range(len(paragraph_tokens))]

                indicator_locations = {"premise":[], "conclusion":[], "neg":[], "turn":[], "and":[]}

                for indicator_type in indicator_types:
                    cur_indicators = locals()[f"{indicator_type}_indicators"]
                    indicator_locations[indicator_type] = locate_indicators_in_sentence(paragraph, paragraph_tokens_noG, cur_indicators, indicators_tokens[indicator_type])
                
                all_indicator_locations = []
                for indicator_type, locations in indicator_locations.items():
                    for location in locations:
                        if location:
                            all_indicator_locations.append((location, lcp_label_mapping[indicator_type]))
                
                all_indicator_locations.sort(key=take_first_idx)
                
                # 保留span最大
                all_indicator_locations = keep_max_span_for_each_start_idx(all_indicator_locations) if all_indicator_locations else all_indicator_locations

                # 添加无indicator mask locations  label=5
                no_logic_mask_idx_candidates = list(range(len(paragraph_tokens_noG)))
                for indicator_location in all_indicator_locations:
                    start_idx, end_idx = indicator_location[0]
                    for i in range(max(0, start_idx-5), min(len(paragraph_tokens_noG) ,end_idx+1+5)):  # 扩大no logic indicator mask禁止mask的范围
                        if i in no_logic_mask_idx_candidates: 
                            no_logic_mask_idx_candidates.remove(i)
                for no_logic_mask_idx_candidate in no_logic_mask_idx_candidates:
                    if random.random() < NOT_LOGIC_INDICATOR_MASKED_RATIO:
                        all_indicator_locations.append(((no_logic_mask_idx_candidate, no_logic_mask_idx_candidate), lcp_label_mapping["no"]))

                # 开始mask, 修改logic_polarity_label
                while all_indicator_locations:
                    cur_location = all_indicator_locations.pop()
                    cur_start_idx, cur_end_idx = cur_location[0]
                    cur_polarity_label = cur_location[1]

                    if cur_polarity_label == lcp_label_mapping["no"]:
                        paragraph_tokens[cur_start_idx] = '[LGMASK]'
                        logic_polarity_label[cur_start_idx] = cur_polarity_label
                        lcp_label_num[cur_polarity_label] += 1
                    else:
                        random_num = random.random()

                        if cur_polarity_label == lcp_label_mapping["and"]:
                            if random_num < (LOGIC_MASKED_RATIO * 0.5):
                                paragraph_tokens[cur_start_idx: cur_end_idx+1] = ['[LGMASK]']
                                logic_polarity_label[cur_start_idx: cur_end_idx+1] = [cur_polarity_label]
                                lcp_label_num[cur_polarity_label] += 1
                        else:
                            if random_num < LOGIC_MASKED_RATIO:
                                paragraph_tokens[cur_start_idx: cur_end_idx+1] = ['[LGMASK]']
                                logic_polarity_label[cur_start_idx: cur_end_idx+1] = [cur_polarity_label]
                                lcp_label_num[cur_polarity_label] += 1


                paragraph = tokenizer.convert_tokens_to_string(paragraph_tokens).replace("[LGMASK]", " [LGMASK]")

                para_tokens_before_save = tokenizer.tokenize(paragraph)

                if len(para_tokens_before_save) > len(logic_polarity_label):
                        logic_polarity_label += [-100 for _ in range(len(para_tokens_before_save) - len(logic_polarity_label))]
                else:
                    logic_polarity_label = logic_polarity_label[:len(para_tokens_before_save)]
                
                if "[LGMASK]" in paragraph:

                    lgmask_indices = []
                    for i in range(len(para_tokens_before_save)):
                        if para_tokens_before_save[i] == "[LGMASK]" : lgmask_indices.append(i)
                    for idx in lgmask_indices:
                        offset = 1
                        if logic_polarity_label[idx] not in label_base:
                            
                            target_idx = None
                            for i in range(1,5):
                                if idx-i > -1 and logic_polarity_label[idx-i] in label_base:
                                    target_idx = idx-i
                                    break
                                if idx+i < len(logic_polarity_label)-1 and logic_polarity_label[idx+i] in label_base:
                                    target_idx = idx+i
                                    break
                            if target_idx:    
                                logic_polarity_label[idx], logic_polarity_label[target_idx] = logic_polarity_label[target_idx], logic_polarity_label[idx]
                            else:
                                is_abandon = True
                                break
                    if is_abandon:
                        is_abandon = False
                        abandoned_sample_num += 1
                        print(f"1 abandoned_sample_num:{abandoned_sample_num}")
                        continue


                    # [LGMASK]验证
                    lgmask_indices = []
                    for i in range(len(para_tokens_before_save)):
                        if para_tokens_before_save[i] == "[LGMASK]" : lgmask_indices.append(i)
                    for idx in lgmask_indices:
                        if logic_polarity_label[idx] not in label_base:
                            is_abandon = True
                    if is_abandon:
                        is_abandon = False
                        abandoned_sample_num += 1
                        print(f"3 abandoned_sample_num:{abandoned_sample_num}")
                        continue

                    if len(tokenizer.tokenize(paragraph)) != len(logic_polarity_label):
                        is_abandon = True
                    if is_abandon:
                        is_abandon = False
                        abandoned_sample_num += 1
                        print(f"4 abandoned_sample_num:{abandoned_sample_num}")
                        continue
                    valid_logic_sample_num += 1

                samples.append(json.dumps({"text": paragraph, "logic_polarity_label": logic_polarity_label}))

            with open(f"{opt_dir}/lcp-rbt-{save_idx}.jsonl", 'w') as opt_f:
                opt_f.write("\n".join(samples))
                samples = []

