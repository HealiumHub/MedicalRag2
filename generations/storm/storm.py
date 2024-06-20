import re
from typing import Tuple
import json

from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

from generations.completion import get_model
from generations.storm.pdf import md_to_pdf
from models.enum import RetrievalApiEnum
from models.types import Source
from langchain.callbacks import get_openai_callback

TOTAL_QUESTIONS = 1
SEARCH_TOP_K = 30
NUM_PERSPECTIVES = 1


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
        system_prompt = f"""You need to select a group of {NUM_PERSPECTIVES} writers who will work together to write a comprehensive article on the topic. Each of them represents a different perspective , role , or affiliation related to this topic .
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
        system_prompt = """You are an experienced writer and want to edit a long-form article about a given topic . Besides your identity as a writer, you have a specific focus when researching the topic.
    Now , you are chatting with an expert to get information . Ask good questions to get more useful information .
    Please ask no more than one question at a time and don't ask what you have asked before. Other than generating the question, don't adding anything additional.
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
        system_prompt = """You are an expert who can use information effectively . You are chatting with a writer who wants to write an article on topic you know .
        You have gathered the related information and will now use the information to form a response.
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
    Use "# Title" to indicate section title , "## Subsection Title" to indicate
    subsection title , "### Sub-subsection title" to indicate sub-subsection title , and so on.
    Do not include other information."""
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
    Use "# Title" to indicate section title , "## Subsection Title" to indicate
    subsection title , "### Sub-subsection title" to indicate sub-subsection title , and so on.
    Do not include other information.\n\ndraft outline: {outline}\n\n"""
        user_prompt = """Learned information: {conversation}"""

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
        a section of a blog and several references. You will generate the article of the section using the provided references.
        If the content can be represented in a table, you can use a table to represent the content.
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
            similarity_top_k=SEARCH_TOP_K,
        )
        return retriever.search([query])

    def generate_conversations(
        self, topic: str, perspectives: list[str]
    ) -> list[list[str]]:
        all_conversations = []
        references = []
        duplicate_references = set()

        for p in perspectives:
            history = []
            for i in range(TOTAL_QUESTIONS):
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
                            "id": result.id,
                            "source": result.file_name,
                            "content": result.content,
                        }
                    )

                answer = self.generate_answer(topic, question, all_context)
                history.append(answer)

            all_conversations.append(history)
        print("DONE CONVERSATION.")
        return all_conversations

    def add_references_in_body_and_remove_invalid_ones(
        self, answer: str, references: list[Source]
    ) -> str:
        # Find all references in the answer. It should be in the form of [ref_id] i.e. [fiejwofij1-foijow2]
        references_in_answer = re.findall(r"\[.*?\]", answer)

        # For each of them, if inside is not a number, replace it with the correct reference.
        reference_id_used = []  # use list to keep the order
        for ref in references_in_answer:
            ref_id = ref[1:-1]

            for reference in references:
                if reference.id == ref_id or (
                    reference.id.startswith(ref_id) and len(ref_id) > 4
                ):
                    if reference.id not in reference_id_used:
                        reference_id_used.append(reference.id)

                    answer = answer.replace(
                        ref, f"[[{len(reference_id_used)}]](#{len(reference_id_used)})"
                    )
                    print(f"Replaced {ref} with {len(reference_id_used)}")
                    break
            else:
                # If it's not found, remove the reference.
                answer = answer.replace(ref, "")
                print(f"Removed {ref}")

        return answer, reference_id_used

    def write_lead_paragraph(self, topic: str, article: str) -> str:
        model = get_model(self.model_name, 0.5, streaming=False)
        system_prompt = """Write an abstract section for the following article. 
It should capture the main idea of the article and provide a brief overview of the content. 
The abstract should be concise and engaging, encouraging the reader to continue reading."""
        user_prompt = """TOPIC: {topic} \n\n ARTICLE: {article}"""
        messages = [
            SystemMessage(system_prompt),
            HumanMessage(user_prompt.format(topic=topic, article=article)),
        ]
        return (model | StrOutputParser()).invoke(messages)

    def write_title_for_article(self, topic: str) -> str:
        model = get_model(self.model_name, 0.5, streaming=False)
        system_prompt = """Write a title for a long-form article about a given topic. 
The title should be catchy and informative.
It should be able to grab the reader's attention and give them an idea of what the article is about.
        """
        user_prompt = """Here's the topic:\n\nTOPIC:{topic}"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt.format(topic=topic)),
        ]
        title = (model | StrOutputParser()).invoke(messages)

        if title.startswith('"'):
            title = title[1:]
        if title.endswith('"'):
            title = title[:-1]

        return title

    def __encode_heading_to_url(self, heading: str) -> str:
        heading = heading.replace("#", "").strip()

        # if it starts with 1. or 1.1. or 1.1.1. etc, remove it.
        regex = r"(^\d+(\.\d+){0,4}\.)"
        if re.match(regex, heading):
            heading = re.sub(regex, "", heading)

        heading = heading.strip()

        return heading.replace(" ", "-").lower()

    def _generate_toc(self, markdown_text: str):
        toc = []
        headings = re.findall(r"^(#+) (.*)$", markdown_text, re.MULTILINE)

        heading: str
        text: str
        for heading, text in headings:
            number_of_hashtags = len(heading) - len(heading.replace("#", ""))

            # With more 1 hashtag, add 2 spaces for each hashtag
            toc.append(
                f"{'  ' * (number_of_hashtags - 1)}- [{text}](#{self.__encode_heading_to_url(text)})"
            )
        return "\n".join(toc)

    def _refine_headings(self, article: str):
        # Clean the headings. Some of the headings have ":" at the end -> remove them
        # Also add numbers to the main headings ("##") only.
        cnt = 1
        for line in article.split("\n"):
            if line.startswith("## "):
                original_line = line
                # Sometime it highlights the heading up. Remove it.
                line = line.replace(line, line.replace(":", ""))
                line = line.replace(line, line.replace("**", ""))

                # Add numbers to the main headings.
                line = line.replace("## ", "")

                # Clean it & add marker so it works on pdfs.
                article = article.replace(
                    original_line,
                    f"<a id='{self.__encode_heading_to_url(line)}'></a>\n## {cnt}. {line}",
                )
                cnt += 1
            # This applies to h3++ headings.
            elif line.startswith("###"):
                # Only add number to the main headings. Subheadings are not numbered, we only clean and add links.
                original_line = line
                # Sometime it highlights the heading up. Remove it.
                line = line.replace(line, line.replace(":", ""))
                line = line.replace(line, line.replace("**", ""))
                article = article.replace(
                    original_line,
                    f"<a id='{self.__encode_heading_to_url(line)}'></a>\n{line}",
                )

        return article

    def add_references_section_and_annotate(
        self, article: str, all_search_results: list[Source]
    ) -> str:
        # Generate references
        article += "\n\n## References\n\n"

        # Dedupe search results
        deduped_search_results: list[Source] = []
        deduped_ids = set()
        for result in all_search_results:
            if result.id not in deduped_ids:
                deduped_ids.add(result.id)
                deduped_search_results.append(result)

        # Sometimes the reference is wrong or is incomplete, so we need to refine it.
        article, reference_id_used = (
            self.add_references_in_body_and_remove_invalid_ones(
                article, deduped_search_results
            )
        )

        # Add the references to the end of the article.
        # ONLY add the references that are used in the article.
        for reference_order, reference_id in enumerate(reference_id_used):
            for result in deduped_search_results:
                if result.id != reference_id:
                    continue

                article += (
                    '<a id="'
                    + str(reference_order + 1)
                    + '"></a>'
                    + f"[{reference_order + 1}] {result.file_name.replace('.pdf', '')}, page {result.page}: {result.content}\n\n"
                )

        debug_references = re.findall(r"\[.*?\)", article)
        print(f"{debug_references=}")

        # Delete all empty references. NOTE: notice the space.
        empty_reference_regex = r"( \[\]\(#.*\))"
        article = re.sub(empty_reference_regex, "", article)

        debug_references = re.findall(r"\[.*?\)", article)
        print(f"{debug_references=}")

        return article

    def write_article(self, topic: str, outline: str = None) -> str:
        with get_openai_callback() as cb:
            related_topics = self.generate_related_topics(topic)
            print(
                f"STEP 1 - generate topics:\n {json.dumps(related_topics, ensure_ascii=False)}"
            )

            # Different perspectives on the topic
            perspectives = self.generate_perspectives(topic, related_topics)
            perspectives = perspectives.split("\n\n")
            print(f"STEP 2 - generate perspectives:\n {perspectives}")

            # The first outline
            if outline is None:
                outline = self.generate_outline(topic)
                print(f"STEP 3 - generate outline: \n {outline}")

            # Refine outline based on conversations
            all_conversations = self.generate_conversations(topic, perspectives)
            refined_outline = self.refine_outline(topic, outline, all_conversations)
            print(f"STEP 4 - Refine outline: \n{refined_outline}")  # Claude

            # Split to new line if it starts with #
            rr = []
            for line in refined_outline.split("\n"):
                if line.startswith("#"):
                    rr.append(line)
                else:
                    rr[-1] += "\n" + line
            # rr = refined_outline.split("\n")
            print(f"{rr=}")

            article = ""
            all_search_results: list[Source] = []

            # First one is title so skip that
            for section_outline in tqdm(rr[1:4]):
                sec, search_result = self.write_section(section_outline)
                all_search_results += search_result
                article += sec + "\n\n"

            with open("raw_article.md", "w") as f:
                f.write(article)

            article = self.add_references_section_and_annotate(
                article, all_search_results
            )

            lead_paragraph = self.write_lead_paragraph(topic, article)
            article = f"## Abstract\n\n{lead_paragraph}\n\n{article}"

            # TOC is here to not contain title, Table of Contents.
            article = self._refine_headings(article)
            toc = self._generate_toc(article)
            article = f"## 0. Table of Contents\n\n{toc}\n\n{article}"

            title = self.write_title_for_article(topic)
            article = f"""
<div style="text-align: right; margin: 0px">
    <img src="assets/rmit-university-logo.png" alt="Logo" style="height:100px;">
</div>

# {title}\n\n{article}
            """

            print("ARTICLE DONE!")
            with open("article.md", "w") as f:
                f.write(article)

            print(cb)

        md_to_pdf(article, "article.pdf")
        return article


# article = Storm().write_article("Mushroom as a diabetes treatment")
