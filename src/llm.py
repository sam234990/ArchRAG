import json
import logging

log = logging.getLogger(__name__)


def get_llm_report(llm_invoker, input_text, max_report_length):
    output = None
    try:
        response = llm_invoker(input_text)
    except Exception as e:

        log.exception("error generating community report")

        output = {}
    return output


if __name__ == "__main__":
    run_llm_report()
