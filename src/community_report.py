from typing import cast
from llm import llm_invoker
from prompts import COMMUNITY_REPORT_PROMPT
from utils import create_arg_parser

def prep_community_report_context():
    pass


def generate_community_report(community_text, args):
    report_prompt = COMMUNITY_REPORT_PROMPT.format(community_text)
    result = llm_invoker(report_prompt, args, max_tokens=args.max_tokens)
    


def community_report():
    reports = []
    prep_community_report_context()
    generate_community_report()
    pass



if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    generate_community_report(args=args)
    pass