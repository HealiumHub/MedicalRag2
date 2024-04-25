import threading
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from const import PromptConfig


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

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text_so_far += token

        # Render
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text_so_far)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs,
    ):
        # Streamlit does not rerender when the answer finishes, so we need to add the disclaimer here.
        self.on_llm_new_token(PromptConfig.DISCLAIMER)
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
