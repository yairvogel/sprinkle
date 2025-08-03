from typing import Any, Callable, Coroutine
from textual.app import App, ComposeResult
from textual.widgets import Header, TextArea, RichLog
from textual import events

Awaitable = Coroutine[Any, Any, None]


class Sprinkle(App):
    on_finish_editing: Callable[[str], Awaitable]

    def __init__(self, text: str, on_finish_editing: Callable[[str], None]):
        super().__init__()
        self.text = text
        self._on_finish_editing = on_finish_editing

    async def on_key(self, event: events.Key) -> None:
        self.query_one(RichLog).write(event)
        if event.key == "enter":
            text = self.query_one(OneLineTextArea).text
            # release control of the terminal without exiting the process
            with self.suspend():
                self._on_finish_editing(text)

    def compose(self) -> ComposeResult:
        yield Header()
        yield OneLineTextArea(self.text, self.on_key)


class OneLineTextArea(TextArea):
    on_return_key: Callable[[events.Key], Awaitable]

    def __init__(self, text: str, on_return_key: Callable[[events.Key], Awaitable]):
        super().__init__(text)
        self.__on_return_key = on_return_key

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            await self.__on_return_key(event)
            event.prevent_default()
            event.stop()
        else:
            await super()._on_key(event)
