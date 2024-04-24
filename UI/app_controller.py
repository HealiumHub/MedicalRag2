from ast import arg
from concurrent.futures import thread
import threading
from responses import target
import streamlit as st

from typing import Optional
from streamlit.runtime.scriptrunner import add_script_run_ctx
from langchain.callbacks.base import BaseCallbackHandler


from const import MODEL_CONTEXT_LENGTH, MODELS
from models.types import Chat, Message, RoleEnum
from retrievals.retrieval import DeepRetrievalApi
from generations.completion import Citation, QuotedAnswer, get_answer_with_context


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
        self.chats = [] if "chats" not in st.session_state else st.session_state.chats
        if "active_chat_idx" not in st.session_state:
            self.active_chat_idx = 0
            active_chat = Chat(id=0)
            self.chats.append(active_chat)
        else:
            self.active_chat_idx = st.session_state.active_chat_idx

    def __render(self):
        self.__render_sidebar()
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
                with st.chat_message(RoleEnum.assistant):
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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method="markdown"):
        self.container = container
        self.display_text = initial_text
        self.display_method = display_method
        self.text_so_far = ""

    def _get_value_of_key_in_streaming_json(
        self, key: str, stream: str
    ) -> Optional[str]:
        """
        Given an incomplete json string, try to grab the content of a key (first order only, not nested).
        This can be used recursively to get nested keys.
        Inputs can be : {"answer": "The answer is 42", "citations": [ ...]}
        OR [{"file_name": "123", "quote_in_source": "The answer is 42", "quote_in_answer": "The answer is 42"}, ...]
        """
        END_QUOTE_MAP = {
            '"': '"',
            "'": "'",
            "{": "}",
            "[": "]",
        }

        first_index = stream.find(f'"{key}": ') + len(f'"{key}": ')
        if first_index == -1:
            return None

        # This can be ' or " or { or [
        opening_quote = stream[first_index]
        if opening_quote != "[":
            opening_quote = '"'

        end_quote = END_QUOTE_MAP[opening_quote]
        end_index = stream.find(end_quote, first_index + 1)
        if end_index == -1:
            end_index = len(stream)

        return stream[first_index:end_index]

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text_so_far += token
        print("debug", self.text_so_far)

        # For structured JSON output, only start display when the key we are looking for is found.
        # it stops when that key's value is finished.
        # {
        #     "answer": "The answer is 42", <-- only show this
        #     "citations": [ ...]
        # }
        
        # NOTE: ONLY raw output for now.
        # answer = self._get_value_of_key_in_streaming_json("answer", self.text_so_far)
        # citation_string = self._get_value_of_key_in_streaming_json(
        #     "citations", self.text_so_far
        # )
        # individual_citations = (
        #     citation_string.split("},") if citation_string is not None else []
        # )
        # print(individual_citations)

        # citation_list = []
        # for i, c in enumerate(individual_citations):
        #     citation = Citation(
        #         file_name=self._get_value_of_key_in_streaming_json("file_name", c),
        #         quote_in_source=self._get_value_of_key_in_streaming_json(
        #             "quote_in_source", c
        #         ),
        #         quote_in_answer=self._get_value_of_key_in_streaming_json(
        #             "quote_in_answer", c
        #         ),
        #     )
        #     citation_list.append(citation)
        #     print(f"Citation {i+1}: {citation}")

        # response_model = QuotedAnswer(answer=answer, citations=citation_list)

        # # The extracted text.
        # self.display_text = str(response_model)

        # Render
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text_so_far)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")
