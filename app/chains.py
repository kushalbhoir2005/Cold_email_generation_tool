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
            You are Kushal, a recent Postgraduate in Data Science & Analytics and an aspiring AI/ML Engineer, Data Analyst, and Business Analyst. 
            You have experience building AI-powered applications, data analytics dashboards, and machine learning solutions using Python, SQL, Power BI, LangChain, Llama, RAG, and related technologies. 
            Also add the most relevant information from the following links to showcase Kushal's portfolio: {link_list}. 
            Remember, you are Kushal, representing your own portfolio and technical expertise.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("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"))