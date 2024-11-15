import requests
import pandas as pd
import re
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from agent import get_random_header, get_proxies
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# 爬虫reference :chatgpt-3.5
def get_wiki_summary_by_title(title):
    # 构建Wikipedia API请求URL
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "exintro": True,  # 获取页面的摘要部分
    }

    # 防止请求太多导致拒绝访问，设置初始设置，reference:https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url-in-requests
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # 发送API请求并获取JSON响应
    response = session.get(
        api_url,
        params=params,
        verify=False,
        headers=get_random_header(),
        proxies=get_proxies(),
    )
    data = response.json()

    # 提取摘要部分
    page_id = list(data["query"]["pages"].keys())[0]
    if "extract" not in data["query"]["pages"][page_id]:
        return None, None
    else:
        summary_html = data["query"]["pages"][page_id]["extract"]
        # 使用BeautifulSoup解析HTML并获取纯文本
        soup = BeautifulSoup(summary_html, "html.parser")
        summary_text = soup.get_text()
        all_summary = summary_text.replace("\n", "")

        paragraphs = summary_text.split("\n")
        first_non_empty_paragraph = next((p for p in paragraphs if p.strip()), None)
        if not first_non_empty_paragraph:
            return None, None
        # 按句子切分
        sentences = re.split(r"(?<=[.!?]) +", first_non_empty_paragraph)
        summary_words = []

        for sentence in sentences:
            words = sentence.split()
            if len(summary_words) + len(words) > 100:
                break
            summary_words.extend(words)

        summary_text = " ".join(summary_words)

    return summary_text, all_summary


def fetch_summary(entity_name):
    return get_wiki_summary_by_title(entity_name)


def get_entity_summary(entity_df, number_workers=32):
    entity_df = entity_df.reset_index(drop=True)

    # 创建一个新的 DataFrame 来存储结果
    summaries = [None] * len(entity_df)  # 初始化与原始 DataFrame 相同长度的列表

    with ThreadPoolExecutor(max_workers=number_workers) as executor:
        future_to_index = {
            executor.submit(fetch_summary, row["entity"]): index
            for index, row in entity_df.iterrows()
        }

        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Fetching summaries"):
            index = future_to_index[future]
            try:
                summary, all_summary = future.result()
                summaries[index] = (summary, all_summary)  # 按索引保存结果
            except Exception as e:
                print(f"Error fetching summary for index {index}: {e}")
                summaries[index] = (None, None)  # 或者其他默认值

    # 将结果添加到 DataFrame
    entity_df[["summary", "all_summary_text"]] = pd.DataFrame(
        summaries, index=entity_df.index
    )

    return entity_df


if __name__ == "__main__":
    entity_ori_path = "/mnt/data/wangshu/hcarag/mintaka/KG/node_ori.csv"
    entity_ori_df = pd.read_csv(entity_ori_path)
    print(entity_ori_df.shape)
    res_df = get_entity_summary(entity_ori_df)
    print(res_df.head())
    print(res_df.tail())
    print(res_df.shape)
    entity_save_path = "/mnt/data/wangshu/hcarag/mintaka/KG/node_summary.csv"
    res_df.to_csv(entity_save_path, index=False)
    
