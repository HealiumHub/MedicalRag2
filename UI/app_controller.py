import logging
import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from const import MODELS, PromptConfig
from generations.completion import get_answer_with_context
from models.enum import RetrievalApiEnum
from models.types import Chat, Message, RoleEnum, Source

from .utilities import ReturnValueThread, StreamHandler

logger = logging.getLogger(__name__)


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

    def __render_chat(self):
        if user_query := st.chat_input("Enter your question here ✍️"):
            with st.chat_message(
                RoleEnum.user, avatar=self.__get_avatar_for_role(RoleEnum.user)
            ):
                st.markdown(user_query)
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
                    )
                )

            with st.spinner("Please wait, I'm searching for references... :eyes:"):
                stop_event = threading.Event()
                thread = ReturnValueThread(
                    target=RetrievalApiEnum.get_retrieval(
                        st.session_state.retrieval_api
                    ).search,
                    args=(user_query,),
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

                for article in related_articles:
                    with st.expander(f"Article {article.id}"):
                        article: Source = article
                        st.markdown(
                            f"""
                            **id**: {article.id}  
                            **DOI**: {article.doi}  
                            **File Name**: {article.file_name}  
                            **Content**: {article.content}  
                            **Score**: {article.score}  
                                    """
                        )

            with st.spinner("I'm thinking..."):
                with st.chat_message(
                    RoleEnum.assistant,
                    avatar=self.__get_avatar_for_role(RoleEnum.assistant),
                ):
                    # Temp chatbox for streaming outputs
                    chat_box = st.markdown("")
                    stream_handler = StreamHandler(chat_box)

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
                                stream_handler,
                            ),
                        )
                        add_script_run_ctx(thread)
                        thread.start()
                        thread.join()
                        stop_event.set()

                        completion = thread.result

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
                            )
                        )
                        st.balloons()
                    except Exception as e:
                        logger.exception(e)
                        completion = "Error happened when generating completion."
                        st.error(completion, icon="🚨")

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
            _ = st.text_area(
                label="Custom instruction",
                value=PromptConfig.PERSONALITY,
                help="Enter a custom instruction to guide the model.",
                key="custom_instruction",
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
