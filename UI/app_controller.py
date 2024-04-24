import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from const import MODELS
from generations.completion import get_answer_with_context
from models.types import Chat, Message, RoleEnum
from retrievals.retrieval import DeepRetrievalApi

from .utilities import ReturnValueThread, StreamHandler


class AppController:
    chats: list[Chat]
    active_chat_idx: int
    selected_model: str

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
        self.chats = st.session_state.chats

        print(st.session_state.to_dict())
        if "active_chat_idx" not in st.session_state:
            st.session_state.active_chat_idx = 0
        self.active_chat_idx = st.session_state.active_chat_idx

    def __render(self):
        self.__render_sidebar()
        self.__render_history()
        self.__render_chat()

    def __get_avatar_for_role(self, role: str) -> str:
        if role == RoleEnum.user:
            return "assets/michael-avt.jpeg"
        elif role == RoleEnum.assistant:
            return "assets/logo.png"

    def __render_history(self):
        print(self.active_chat_idx)
        st.header("How can I help you today? 😄", divider="rainbow")
        for message in self.chats[self.active_chat_idx].messages:
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

            with st.spinner("Please wait, I'm searching for references... :eyes:"):
                stop_event = threading.Event()
                thread = ReturnValueThread(
                    target=DeepRetrievalApi().search, args=(user_query,)
                )
                add_script_run_ctx(thread)
                thread.start()
                thread.join()
                stop_event.set()

                try:
                    related_articles = thread.result
                except Exception as e:
                    related_articles = [], ""
                    st.error("Error happened when searching for docs.")

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
                                self.selected_model,
                                related_articles,
                                stream_handler,
                            ),
                        )
                        add_script_run_ctx(thread)
                        thread.start()
                        thread.join()
                        stop_event.set()

                        completion = thread.result

                        # Save the response to history.
                        self.chats[self.active_chat_idx].messages.append(
                            Message(
                                id=len(self.chats[self.active_chat_idx].messages) + 1,
                                role=RoleEnum.assistant,
                                content=completion,
                            )
                        )
                    except Exception as e:
                        completion = "Error happened when generating completion."
                        st.error(completion)
                    st.balloons()

    def __render_sidebar(self):
        with st.sidebar:
            st.title("HealthLight ⛑️")
            st.markdown(
                """
                This is a project that aims to provide a conversational AI system that can help you with your health-related questions.
                """
            )
            st.divider()
            self.selected_model = st.selectbox(
                label="Choose LLM model",
                options=MODELS,
                label_visibility="collapsed",
            )
            # TODO Update the selected_model

            print("index", self.active_chat_idx)
            
            # TODO: Dropdown is sus af.
            selected_chat = st.selectbox(
                label="Chats",
                options=[chat.id for chat in self.chats],
                index=self.active_chat_idx,
                format_func=lambda x: f"Chat {x + 1}",
                label_visibility="collapsed",
            )
            if selected_chat is None:
                selected_chat = self.active_chat_idx

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
        self.__change_thread(active_chat.id)

    def __change_thread(self, idx: int):
        self.active_chat_idx = idx
        st.session_state.active_chat_idx = idx
