import streamlit as st

from UI.const import MODELS
from completion import QuotedAnswer, get_answer_with_context



class AppController:
    def __init__(self):
        self.__config()
        self.__render()

    def __config(self):
        if "messages_dict" not in st.session_state:
            st.session_state.messages_dict = {}
        if "related_articles" not in st.session_state:
            st.session_state.related_articles = {}
        # Init thread and message
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = 1
            st.session_state.thread_ids.append(st.session_state.thread_id)
            st.session_state.messages = st.session_state.messages_dict.get(
                st.session_state.thread_id, []
            )
        # Initialize messages if not present.
        if "messages" not in st.session_state:
            st.session_state.messages = st.session_state.messages_dict.get(
                st.session_state.thread_id, []
            )

    def __render(self):
        st.set_page_config(
            page_title="HealthLight Project",
            page_icon="⛑️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Select model
        selected_model = st.sidebar.selectbox("Choose completion model", MODELS)

        # New threads
        st.sidebar.divider()
        if st.sidebar.button(label="New thread", type="primary", use_container_width=True):
            st.session_state.thread_id = len(st.session_state.thread_ids) + 1
            st.session_state.thread_ids.append(st.session_state.thread_id)
            st.session_state.messages = st.session_state.messages_dict.get(
                st.session_state.thread_id, []
            )
        # Update thread messages.
        st.session_state.messages_dict[
            st.session_state.thread_id
        ] = st.session_state.messages

        for i, thread_id in enumerate(st.session_state.thread_ids):
            if thread_id == st.session_state.thread_id:
                st.sidebar.button(
                    f"**Thread {i + 1} :point_left:**",
                    on_click=self.change_thread,
                    args=(thread_id,),
                    use_container_width=True,
                )
            else:
                st.sidebar.button(
                    f"Thread {i + 1}",
                    on_click=self.change_thread,
                    args=(thread_id,),
                    use_container_width=True,
                )
        # Title
        left_col, right_col = st.columns([0.5, 0.5], gap="medium")
        left_col.markdown("### MedLight")
        right_col.markdown("### References")

        if len(st.session_state.related_articles[st.session_state.thread_id]) > 0:
            right_col.markdown(
                "\n".join(
                    [
                        f'##### [{i + 1}. DOI {x["file_name"]}, {x["file_name"]}]()'
                        + f"\n {self.format_content(x)}\n***"
                        for i, x in enumerate(
                        st.session_state.related_articles[st.session_state.thread_id]
                    )
                    ]
                )
            )
        else:
            pass

        # Render the messages
        for msg in st.session_state.messages:
            role = msg.role if not isinstance(msg, dict) else msg["role"]
            content: str = (
                msg.content[0].text.value
                if not isinstance(msg, dict)
                else msg["content"][0]["text"]["value"]
            )

            if content is None:
                continue

            if isinstance(content, QuotedAnswer):
                # Render reference buttons.
                # TODO: Need better UI for this one.
                with left_col.chat_message(role):
                    for i, citation in enumerate(content.citations):
                        st.button(f"[{i+1}] from {citation.source_id}")

                content = str(content)

            # print(f"Role: {role}, Content: {content}")

            # Show disclaimer for assistant messages.
            if (
                    role == "assistant"
                    and "Please consult professional doctor for accurate medical advices."
                    not in content
            ):
                content += (
                    "\n\n*Please consult professional doctor for accurate medical advices.*"
                )

            if role == "system":  # Don't show system messages.
                pass
            else:
                with left_col.chat_message(role):
                    st.markdown(content)
    def change_thread(self, thread_id):
        st.session_state.thread_id = thread_id
        st.session_state.messages = st.session_state.messages_dict.get(
            st.session_state.thread_id, []
        )

    def format_content(self, x):
        return x["content"][:500] + "..." if len(x["content"]) > 500 else x["content"]
