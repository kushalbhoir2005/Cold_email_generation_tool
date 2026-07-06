import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
    temperature=0, 
    groq_api_key='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', 
     model_name="llama-3.3-70b-versatile"
)
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
             You are Kushal, a AI Engineer at Probus Insurance Broker Pvt Ltd.Probus is an InsurTech company dedicated to facilitating 
        the seamless comparison, purchase, and renewal of insurance plans through automated digital tools. 
        Over our experience, we have empowered numerous policyholders and 
        advisors with tailored solutions, fostering scalability, process optimization, cost reduction, and heightened overall efficiency.
            Also add the most relevant ones from the following links to showcase Probus portfolio: {link_list}
            Remember you are Kushal, AI Engineer at Probus.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("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"))