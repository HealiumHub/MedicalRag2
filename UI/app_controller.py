import threading
import streamlit as st

from UI.const import MODELS


class AppController:
    def __init__(self):
        self.__config()
        self.__render()

    def __config(self):
        st.set_page_config(
            page_title="HealthLight Project",
            page_icon="⛑️",
            layout="wide",
        )

        if "threads" not in st.session_state:
            st.session_state.threads = []
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = 0
            st.session_state.threads.append(st.session_state.thread_id)

    def __render(self):
        self.__render_sidebar()
        self.__render_main()

    def __render_main(self):
        st.header("How can I help you today? 😄", divider="rainbow")
        for message in st.session_state.get("messages", []):
            role = message["role"]
            content = message["content"]
            if content is None:
                continue
            if role == "system":
                pass
            if role == "assistant":
                content += "\n\n*Please consult professional doctor for accurate medical advices.*"
            with st.chat_message(
                role,
                avatar=(
                    "assets/michael-avt.jpeg" if role == "user" else "assets/logo.png"
                ),
            ):
                st.markdown(content)
        if user_query := st.chat_input("Enter your question here"):
            with st.chat_message("user", avatar=f"assets/michael-avt.jpeg"):
                st.markdown(user_query)
                st.session_state.messages.append(
                    {"role": "user", "content": user_query}
                )

                # with st.spinner("Searching for the references..."):
                #     stop_event = threading.Event()
                # with st.spinner("Generating answer..."):
                #     with st.chat_message("assistant", avatar=st.image("assets/logo.png")):

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
            selected_chat = st.selectbox(
                label="Threads",
                options=st.session_state.get("threads", []),
                index=st.session_state.thread_id,
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
        st.session_state.thread_id = len(st.session_state.get("threads", []))
        st.session_state.threads = st.session_state.get("threads", [])
        st.session_state.threads.append(st.session_state.thread_id)

    def __change_thread(self, thread_id: int):
        st.session_state.thread_id = thread_id


class ReturnValueThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        self.result = self._target(*self._args, **self._kwargs)
