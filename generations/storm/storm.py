from typing import Tuple
import json

from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

from generations.completion import get_model
from models.enum import RetrievalApiEnum
from models.types import Source


class Storm:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        self.model_name = model_name
        self.temperature = temperature

    def generate_related_topics(self, topic: str) -> list[str]:
        model = get_model(self.model_name, 0.5, streaming=False)
        system_prompt = """I want to write a long-form article about a topic. I will give you the topic and I want you to suggest 3 related sub-topics to expand the content."""
        user_prompt = """Here's the topic:\n\nTOPIC:{topic}"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt.format(topic=topic)),
        ]

        return (model | StrOutputParser()).invoke(messages)

    def generate_perspectives(self, topic: str, related_topics: str) -> list[str]:
        model = get_model(self.model_name, 0.5, streaming=False)
        system_prompt = """You need to select a group of 3 writers who will work together to write a comprehensive article on the topic. Each of them represents a different perspective , role , or affiliation related to this topic .
        You can use other related topics for inspiration. For each role, add description of what they will focus on. Give your answer strictly in the following format without adding anything additional:1. short summary of writer one: description \n 2. short summary of writer two: description \n...\n\n"""
        user_prompt = (
            """Here's the topic:\n\nTOPIC:{topic}\n\nRelated topics: {related_topics}"""
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=user_prompt.format(topic=topic, related_topics=related_topics)
            ),
        ]

        return (model | StrOutputParser()).invoke(messages)

    def generate_question(self, topic: str, perspective: str, history: list[str]):
        model = get_model(self.model_name, 0.5, streaming=False)
        system_prompt = """You are an experienced writer and want to edit a long-form article about a given topic . Besides your identity as a writer, you have a specific focus when researching the topic .
    Now , you are chatting with an expert to get information . Ask good questions to get more useful information .
    Please ask no more than one question at a time and don 't ask what you have asked before. Other than generating the question, don't adding anything additional.
    Your questions should be related to the topic you want to write.\n\nConversation history: {history}\n\n"""
        user_prompt = """Here's the topic:\n\nTOPIC:{topic}\n\nYour specific focus: {perspective}\n\nQuestion:"""

        context = "\n".join(history)
        messages = [
            SystemMessage(content=system_prompt.format(history=context)),
            HumanMessage(
                content=user_prompt.format(topic=topic, perspective=perspective)
            ),
        ]
        return (model | StrOutputParser()).invoke(messages)

    def generate_answer(self, topic: str, question: str, context: str):
        model = get_model(self.model_name, 0.5, streaming=False)
        system_prompt = """You are an expert who can use information effectively . You are chatting with a
    writer who wants to write an article on topic you know . You
    have gathered the related information and will now use the information to form a response.
    Make your response as informative as possible and make sure every sentence is supported by the gathered information.\n\nRelated information: {context}\n\n"""
        user_prompt = """Here's the topic:\n\nTOPIC:{topic}\n\nQuestion: {question}"""
        messages = [
            SystemMessage(content=system_prompt.format(context=context)),
            HumanMessage(content=user_prompt.format(topic=topic, question=question)),
        ]

        return (model | StrOutputParser()).invoke(messages)

    def generate_outline(self, topic: str) -> str:
        system_prompt = """Write an outline for an article about a given topic.
    Here is the format of your writing:
    Use "#" Title " to indicate section title , "##" Title " to indicate subsection title , "###" Title " to indicate subsubsection title , and so on.
    Do not include other information.\n\n"""
        user_prompt = """Here's the topic:\n\nTOPIC:{topic}"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt.format(topic=topic)),
        ]
        model = get_model(self.model_name, 0.5, streaming=False)
        return (model | StrOutputParser()).invoke(messages)

    def refine_outline(
        self, topic: str, outline: str, conversation: list[list[str]]
    ) -> str:
        model = get_model(self.model_name, 0.5, streaming=False)
        system_prompt = """I want you to improve an outline of an article about {topic} topic. You already have a draft outline given below that
    covers the general information. Now you want to improve it based on the given
    information learned from an information - seeking conversation to make it more
    comprehensive. Here is the format of your writing:
    Use "#" Title " to indicate section title , "##" Title " to indicate
    subsection title , "###" Title " to indicate subsubsection title , and so on. Do not include other information.\n\ndraft outline: {outline}\n\n"""
        user_prompt = """learned information: {conversation}"""

        flattened_list = [item for sublist in conversation for item in sublist]
        context: str = "".join(flattened_list)

        messages = [
            SystemMessage(system_prompt.format(topic=topic, outline=outline)),
            HumanMessage(user_prompt.format(conversation=context)),
        ]
        return (model | StrOutputParser()).invoke(messages)

    def generate_references_string(self, references: list[Source]) -> str:
        output = []
        for ref in references:
            ref_id = ref.id
            ref_url = ref.file_name
            ref_content = ref.content

            # Construct a formatted string for each reference
            reference_str = (
                f"Reference ID:\n {ref_id}\nURL: {ref_url}\nContent: {ref_content}\n"
            )

            output.append(reference_str)

        return "\n".join(output)

    def write_section(self, section: str) -> Tuple[str, list[Source]]:
        model = get_model(self.model_name, 0.5, streaming=False)
        search_result = self.search(section)
        references = self.generate_references_string(search_result)

        system_prompt = """You are an expert in writing. I will give you an outline of
        a section of a blog and several references. You will generate the article of the section using the provided refrences.
        You MUST cite your writing using the given sources. Do not include other information. Include 'reference id' for each sentence in this format: [ref_id]. Your response MUST be in markdown format.\n\nREFERENCES: {references}\n\n"""
        user_prompt = """SECTION OUTLINE: {section}"""
        messages = [
            SystemMessage(system_prompt.format(references=references)),
            HumanMessage(user_prompt.format(section=section)),
        ]
        return (model | StrOutputParser()).invoke(messages), search_result

    def search(self, query: str) -> list[Source]:
        retriever = RetrievalApiEnum.get_retrieval(
            RetrievalApiEnum.CHROMA_RETRIEVAL,
            similarity_top_k=50,
        )
        return retriever.search([query])

    def generate_conversations(
        self, topic: str, perspectives: list[str]
    ) -> list[list[str]]:
        all_conversations = []
        references = []
        duplicate_references = set()
        total_questions = 3

        for p in perspectives[:1]:
            history = []
            for i in range(total_questions):
                question = self.generate_question(topic, p, history)
                print(f"QUESTION {i}: {question}")

                history.append(question)
                search_results: list[Source] = self.search(question)
                all_context = ""

                for result in search_results:
                    all_context += result.content + "\n"
                    if result.id in duplicate_references:
                        continue

                    duplicate_references.add(result.id)
                    references.append(
                        {
                            "title": result.file_name,
                            "source": result.file_name,
                            "content": result.content,
                        }
                    )

                answer = self.generate_answer(topic, question, all_context)
                history.append(answer)

            all_conversations.append(history)
        print("DONE CONVERSATION.")
        return all_conversations

    def write_article(self, topic: str) -> str:
        related_topics = self.generate_related_topics(topic)
        print(f"STEP 1 - generate topics:\n {json.dumps(related_topics)}")

        # Different perspectives on the topic
        perspectives = self.generate_perspectives(topic, related_topics)
        perspectives = perspectives.split("\n\n")
        print(f"STEP 2 - generate perspectives:\n {perspectives}")

        # The first outline
        outline = self.generate_outline(topic)
        print(f"STEP 3 - generate outline: \n {outline}")

        # Refine outline based on conversations
        all_conversations = self.generate_conversations(topic, perspectives)
        refined_outline = self.refine_outline(topic, outline, all_conversations)
        print(f"STEP 4 - Refine outline: \n{refined_outline}")  # Claude

        rr = refined_outline.split("\n\n")
        print(f"{rr=}")

        article = ""
        all_search_results = []
        for section_outline in tqdm(rr[1::]):
            sec, search_result = self.write_section(section_outline)
            all_search_results += search_result
            article += sec + "\n\n"
            
        # Dedupe search results
        deduped_search_results: list[Source]= []
        deduped_ids = set()
        for result in all_search_results:
            if result.id not in deduped_ids:
                deduped_ids.add(result.id)
                deduped_search_results.append(result)
        
        # Generate references
        article += "\n\n#References\n\n"
        for i, result in enumerate(deduped_search_results):
            article = article.replace(f"[{result.id}]", f"[{i + 1}]")
            article += f"[{i + 1}] {result.id} - {result.file_name}: {result.content}\n\n"

        print("ARTICLE DONE!")
        with open("article.md", "w") as f:
            f.write(article)
        return article


article = Storm().write_article(
    "Mushroom as a diabetes treament"
)
