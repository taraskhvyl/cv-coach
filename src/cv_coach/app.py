"""Gradio UI. Run with `python -m cv_coach.app`."""
import gradio as gr
from dotenv import load_dotenv

load_dotenv()  # must run before the OpenAI/Langfuse clients are created


from langfuse import observe  # noqa: E402

from . import coach, voice  # noqa: E402
from .rag import Index, load_text  # noqa: E402


def build_index(path):
    return Index(load_text(path)) if path else None


@observe(name="chat-turn")
def respond(text, audio, history, index, jd):
    if index is None:
        raise gr.Error("Upload your CV first.")
    question = voice.transcribe(audio) if audio else text
    if not question:
        raise gr.Error("Type or record a question.")
    plain = [{"role": m["role"], "content": str(m["content"])} for m in history]
    reply = coach.answer(question, index, jd, plain)
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
    return history, "", None, voice.speak(reply) if audio else None  # spoken reply only for voice questions


with gr.Blocks(title="CV Coach") as demo:
    gr.Markdown("# CV Coach\nUpload your CV, optionally paste a job description, then ask by text or voice.")
    index = gr.State()
    with gr.Row():
        cv = gr.File(label="CV (.pdf / .txt)", file_types=[".pdf", ".txt"], type="filepath")
        jd = gr.Textbox(label="Job description (optional)", lines=6)
    chat = gr.Chatbot(height=400)
    text = gr.Textbox(label="Question", placeholder="How do I tailor my CV to this role?")
    mic = gr.Audio(sources=["microphone"], type="filepath", label="Or ask by voice")
    spoken = gr.Audio(label="Spoken reply", autoplay=True)
    cv.change(build_index, cv, index)
    inputs, outputs = [text, mic, chat, index, jd], [chat, text, mic, spoken]
    text.submit(respond, inputs, outputs)
    mic.stop_recording(respond, inputs, outputs)


def main():
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
