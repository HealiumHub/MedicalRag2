import logging
import re
import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from const import MODELS, PromptConfig
from generations.completion import get_answer_with_context
from models.enum import RetrievalApiEnum
from models.types import Chat, Message, RoleEnum, Source
from postretrieve.rerank import Reranker
from preretrieve.expansion.langchain.expansion import QueryExpansion
from preretrieve.hyde import HyDE

from .utilities import ReturnValueThread, StreamHandler
from generations.translate.detect import LanguageEnum, get_language_of_text
from generations.translate.translate import Translator

logger = logging.getLogger(__name__)

# Cache it so model don't get loaded again
reranker = Reranker()
translator = Translator()


class AppController:
    def __init__(self):
        self.__config()
        self.__render()

    def __config(self):
        st.set_page_config(
            page_title="HealthLight Project",
            page_icon="⛑️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Initialize the chat history & active chat index.
        if "chats" not in st.session_state:
            st.session_state.chats = [Chat(id=0)]

        if "active_chat_idx" not in st.session_state:
            st.session_state.active_chat_idx = 0

        if "selected_model" not in st.session_state:
            st.session_state.selected_model = 0

    def __render(self):
        self.__render_sidebar()
        self.__render_history()
        self.__render_chat()

    def __get_avatar_for_role(self, role: str) -> str:
        if role == RoleEnum.user:
            return "assets/michael-avt.jpeg"
        # elif role == RoleEnum.assistant:
        else:
            return "assets/logo.png"

    def __render_history(self):
        logger.info(st.session_state.active_chat_idx)
        st.header("How can I help you today? 😄", divider="rainbow")

        message: Message
        for message in st.session_state.chats[
            st.session_state.active_chat_idx
        ].messages:
            role = message.role
            content = message.content

            if content is None:
                continue
            if role == RoleEnum.system:
                pass

            with st.chat_message(
                role,
                avatar=(self.__get_avatar_for_role(role)),
            ):
                st.markdown(content)
                self.__render_expended_queries(
                    message.expanded_queries, message.hyde_passages
                )
                self.__render_references(message.related_articles)

    def __render_chat(self):
        if user_query := st.chat_input("Enter your question here ✍️"):
            with st.chat_message(
                RoleEnum.user, avatar=self.__get_avatar_for_role(RoleEnum.user)
            ):
                st.markdown(user_query)

                # Preretrieval = Query Expansion + HyDE
                with st.spinner(
                    "Please wait, I'm analyzing your questions... :mag_right:"
                ):
                    query_language = get_language_of_text(user_query)
                    if query_language == LanguageEnum.VIETNAMESE.value:
                        user_query = translator.translate(user_query)
                        logger.info(f"Translated query: {user_query}")

                    queryExpansion = QueryExpansion(with_openAI=True)

                    # Default is only user query
                    expanded_queries = [user_query]
                    if st.session_state.use_query_expansion:
                        expanded_queries = queryExpansion.paraphase_query(user_query)

                    # Default is no hyde_passages
                    hyde_passages = expanded_queries
                    if st.session_state.use_hyde:
                        hyde_passages = [
                            HyDE().run(query) for query in expanded_queries
                        ]

                    # Only render this if either query expansion or HyDE is enabled.
                    if (
                        st.session_state.use_query_expansion
                        or st.session_state.use_hyde
                    ) and query_language == LanguageEnum.ENGLISH.value:
                        # NOTE: we hide it if the query is in Vietnamese. We don't want to show the user the translated version.
                        self.__render_expended_queries(expanded_queries, hyde_passages)

                st.session_state.chats[
                    st.session_state.active_chat_idx
                ].messages.append(
                    Message(
                        id=len(
                            st.session_state.chats[
                                st.session_state.active_chat_idx
                            ].messages
                        )
                        + 1,
                        role=RoleEnum.user,
                        content=user_query,
                        expanded_queries=expanded_queries,
                        hyde_passages=hyde_passages,
                        related_articles=[],
                    )
                )

            with st.chat_message(
                RoleEnum.assistant,
                avatar=self.__get_avatar_for_role(RoleEnum.assistant),
            ):
                with st.spinner("Please wait, I'm searching for references... :eyes:"):
                    stop_event = threading.Event()
                    thread = ReturnValueThread(
                        target=RetrievalApiEnum.get_retrieval(
                            retrieval_type=st.session_state.retrieval_api,
                            alpha=st.session_state.sparse_dense_weight,
                            similarity_top_k=st.session_state.similarity_top_k,
                        ).search,
                        args=(hyde_passages,),
                    )
                    add_script_run_ctx(thread)
                    thread.start()
                    thread.join()
                    stop_event.set()

                    try:
                        related_articles: list[Source] = thread.result
                    except Exception as e:
                        related_articles = []
                        st.error("Error happened when searching for docs.", icon="🚨")

                # Post retrieval steps
                with st.spinner("I'm organizing the articles..."):
                    reranked_articles = []
                    if st.session_state.use_rerank:
                        # Rerank the articles

                        # TODO: Cache the reranker?
                        reranked_articles = reranker.get_top_k(
                            user_query, related_articles, st.session_state.rerank_top_k
                        )
                        related_articles = reranked_articles

                with st.spinner("I'm thinking..."):
                    # Temp chatbox for streaming outputs
                    chat_box = st.markdown("")
                    stream_handler = StreamHandler(chat_box)

                    # If it's Vietnamese, first get the translation, then translate and stream it back.
                    try:
                        stop_event = threading.Event()
                        thread = ReturnValueThread(
                            target=get_answer_with_context,
                            args=(
                                user_query,
                                st.session_state.selected_model,
                                related_articles,
                                st.session_state.custom_instruction,
                                st.session_state.temperature,
                                stream_handler
                                if query_language == LanguageEnum.ENGLISH.value
                                else None,
                            ),
                        )
                        add_script_run_ctx(thread)
                        thread.start()
                        thread.join()
                        stop_event.set()

                        completion = thread.result

                        # Translate and stream back the completion.
                        if query_language == LanguageEnum.VIETNAMESE.value:
                            logger.info(f"Streaming translated response: {completion}")
                            # Translate it back to Vietnamese.
                            completion = translator.translate(
                                completion,
                                target_language=LanguageEnum.VIETNAMESE.value,
                                stream_handler=stream_handler,
                            )

                        self.__render_references(related_articles, reranked_articles)

                        # Save the response to history.
                        st.session_state.chats[
                            st.session_state.active_chat_idx
                        ].messages.append(
                            Message(
                                id=len(
                                    st.session_state.chats[
                                        st.session_state.active_chat_idx
                                    ].messages
                                )
                                + 1,
                                role=RoleEnum.assistant,
                                content=completion,
                                expanded_queries=[],
                                hyde_passages=[],
                                related_articles=related_articles,
                                reranked_articles=reranked_articles,
                            )
                        )
                        st.balloons()
                    except Exception as e:
                        logger.exception(e)
                        completion = "Error happened when generating completion."
                        st.error(completion, icon="🚨")

    def __render_expended_queries(
        self, expanded_queries: list[str], hyde_passages: list[str]
    ):
        st.markdown(
            "**Let's Get Curious Together! These are questions that we ask on your behalf.**"
        )
        for query, passage in zip(expanded_queries, hyde_passages):
            with st.expander(
                f":mag_right: {query}",
                expanded=True,
            ):
                st.markdown(f"""{passage}""")

    def __render_references(
        self, related_articles: list[Source], reranked_articles: list[Source] = []
    ):
        # for article in related_articles:
        #     with st.expander(f"Article {article.id}"):
        #         st.markdown(
        #             f"""
        #                             **id**: {article.id}
        #                             **DOI**: {article.doi}
        #                             **File Name**: {article.file_name}
        #                             **Page**: {article.page}
        #                             **Content**: {article.content}
        #                             **Score**: {article.score}
        #                                     """
        #         )

        for article in reranked_articles:
            with st.expander(f"Reranked Article {article.id}"):
                st.markdown(
                    f"""
                                    **id**: {article.id}  
                                    **DOI**: {article.doi}  
                                    **File Name**: {article.file_name}  
                                    **Content**: {article.content}  
                                    **Score**: {article.score}  
                                            """
                )

    def __render_sidebar(self):
        with st.sidebar:
            st.title("HealthLight ⛑️")
            st.markdown(
                """
                This is a project that aims to provide a conversational AI system that can help you with your health-related questions.
                """
            )
            st.divider()
            st.session_state.selected_model = st.selectbox(
                label="Choose LLM model",
                options=MODELS,
                label_visibility="collapsed",
            )
            # TODO Update the selected_model

            logger.info(
                f"{st.session_state.active_chat_idx=}, {st.session_state.selected_model=}"
            )

            st.session_state.selected_chat = st.selectbox(
                label="Chats",
                options=[chat.id for chat in st.session_state.chats],
                index=st.session_state.active_chat_idx,
                format_func=lambda x: f"Chat {x + 1}",
                label_visibility="collapsed",
            )

            st.button(
                label="New Chat",
                type="primary",
                on_click=self.__new_chat,
                use_container_width=True,
            )

            st.divider()
            _ = st.selectbox(
                "Choose Retrieval API",
                options=[api_name for api_name in RetrievalApiEnum.__members__],
                key="retrieval_api",
            )

            # Choose number of chunks to retrieve
            _ = st.slider(
                label="Number of chunks to retrieve from retrieval module",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                format="%d",
                key="similarity_top_k",
            )

            st.divider()
            # Toggle to use rerank
            _ = st.checkbox(
                label="Use rerank",
                value=True,
                key="use_rerank",
            )

            # Choose top-k chunks after post-retrieval
            _ = st.slider(
                label="Number of chunks to take after rerank",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                format="%d",
                key="rerank_top_k",
            )
            st.divider()

            # Toggle to use query expansion and HyDE
            _ = st.checkbox(
                label="Use Query Expansion",
                value=True,
                key="use_query_expansion",
            )
            _ = st.checkbox(
                label="Use HyDE",
                value=True,
                key="use_hyde",
            )

            _ = st.text_area(
                label="Custom instruction",
                value=PromptConfig.PERSONALITY,
                help="Enter a custom instruction to guide the model.",
                key="custom_instruction",
            )
            _ = st.slider(
                label="Lexical/Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                format="%f",
                help="Weight for sparse/dense retrieval, only used for hybrid query mode. (0 = lexical, 1 = semantic)",
                key="sparse_dense_weight",
            )
            _ = st.slider(
                label="Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.05,
                format="%f",
                help="The temperature of the sampling distribution.",
                key="temperature",
            )

    def __new_chat(self):
        active_chat = Chat(id=len(st.session_state.chats))
        st.session_state.chats.append(active_chat)
        st.session_state.active_chat_idx = active_chat.id
