import json
import logging
from prompts import COMMUNITY_REPORT_PROMPT


log = logging.getLogger(__name__)




def get_llm_report(llm_invoker, input_text, max_report_length):
    output = None
    try:
        report_prompt = COMMUNITY_REPORT_PROMPT.format(input_text)

        response = llm_invoker(report_prompt)
        output = response.json() or {}
    except Exception as e:

        log.exception("error generating community report")

        output = {}
    return output


if __name__ == "__main__":
    get_llm_report()
