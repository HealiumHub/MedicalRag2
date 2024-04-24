import threading
import streamlit as st

from typing import Optional
from streamlit.runtime.scriptrunner import add_script_run_ctx
from langchain.callbacks.base import BaseCallbackHandler


from const import MODELS
from models.types import Chat, Message, RoleEnum
from retrievals.retrieval import DeepRetrievalApi
from generations.completion import Citation, QuotedAnswer, get_answer_with_context


class AppController:
    chats: list[Chat]
    active_chat_idx: int

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
        self.chats = [] if "chats" not in st.session_state else st.session_state.chats
        if "active_chat_idx" not in st.session_state:
            self.active_chat_idx = 0
            active_chat = Chat(id=0)
            self.chats.append(active_chat)
        else:
            self.active_chat_idx = st.session_state.active_chat_idx

    def __render(self):
        self.__render_sidebar()
        self.__render_main()

    def __render_main(self):
        self.__render_history()
        self.__render_chat()

    def __render_history(self):
        st.header("How can I help you today? 😄", divider="rainbow")
        for message in self.chats[self.active_chat_idx].messages:
            role = message.role
            content = message.content
            if content is None:
                continue
            if role == RoleEnum.system:
                pass
            if role == RoleEnum.assistant:
                content += "\n\n*Please consult professional doctor for accurate medical advices.*"
            with st.chat_message(
                role,
                avatar=(
                    "assets/michael-avt.jpeg"
                    if role == RoleEnum.user
                    else "assets/logo.png"
                ),
            ):
                st.markdown(content)

    def __render_chat(self):
        if user_query := st.chat_input("Enter your question here ✍️"):
            with st.chat_message(RoleEnum.user, avatar=f"assets/michael-avt.jpeg"):
                st.markdown(user_query)
                self.chats[self.active_chat_idx].messages.append(
                    Message(
                        id=len(self.chats[self.active_chat_idx].messages) + 1,
                        role=RoleEnum.user,
                        content=user_query,
                    )
                )
                st.session_state.chats = self.chats

    def __render_sidebar(self):
        with st.sidebar:
            st.title("HealthLight ⛑️")
            st.markdown(
                """
                This is a project that aims to provide a conversational AI system that can help you with your health-related questions.
                """
            )
            st.divider()
            selected_model = st.selectbox(
                label="Choose LLM model",
                options=MODELS,
                label_visibility="collapsed",
            )
            # TODO Update the selected_model

            selected_chat = st.selectbox(
                label="Chats",
                options=[chat.id for chat in self.chats],
                index=self.active_chat_idx,
                format_func=lambda x: f"Chat {x + 1}",
                label_visibility="collapsed",
            )
            self.__change_thread(selected_chat)
            st.button(
                label="New Chat",
                type="primary",
                on_click=self.__new_chat,
                use_container_width=True,
            )

    def __new_chat(self):
        active_chat = Chat(id=len(self.chats))
        self.chats.append(active_chat)
        st.session_state.chats = self.chats
        st.session_state.active_chat_idx = active_chat.id

    def __change_thread(self, idx: int):
        self.active_chat_idx = idx


class ReturnValueThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        self.result = self._target(*self._args, **self._kwargs)
