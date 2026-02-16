import os
import textwrap
import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


st.set_page_config(
	page_title="Retro Quiz Forge",
	page_icon="🕹️",
	layout="wide",
)


def _css() -> str:
	return """
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

	:root {
		--retro-bg: radial-gradient(circle at 10% 20%, #1a0033 0%, #0b001a 45%, #05000d 100%);
		--grid-color: rgba(0, 255, 204, 0.08);
		--neon-pink: #ff4fd8;
		--neon-cyan: #00ffd1;
		--neon-yellow: #ffe66d;
		--neon-purple: #7d5cff;
		--panel-bg: rgba(8, 8, 20, 0.78);
		--panel-border: rgba(0, 255, 204, 0.35);
		--text-main: #e9fbff;
		--text-dim: #9bc7d1;
	}

	html, body, [class*="css"], .stApp {
		background: var(--retro-bg);
		color: var(--text-main);
		font-family: 'VT323', monospace;
		letter-spacing: 0.4px;
	}

	.stApp::before {
		content: "";
		position: fixed;
		inset: 0;
		background-image:
			linear-gradient(to right, var(--grid-color) 1px, transparent 1px),
			linear-gradient(to bottom, var(--grid-color) 1px, transparent 1px);
		background-size: 28px 28px;
		opacity: 0.7;
		pointer-events: none;
		z-index: 0;
	}

	.block-container {
		max-width: 1200px;
		padding-top: 1.5rem;
		position: relative;
		z-index: 1;
	}

	.retro-header {
		border: 2px solid var(--panel-border);
		border-radius: 16px;
		padding: 24px;
		background: linear-gradient(135deg, rgba(255, 79, 216, 0.12), rgba(0, 255, 209, 0.1));
		box-shadow: 0 0 24px rgba(0, 255, 209, 0.2), inset 0 0 18px rgba(255, 79, 216, 0.15);
	}

	.retro-title {
		font-family: 'Press Start 2P', cursive;
		font-size: 28px;
		color: var(--neon-yellow);
		text-shadow: 0 0 12px rgba(255, 230, 109, 0.6), 0 0 24px rgba(255, 79, 216, 0.45);
		margin-bottom: 10px;
	}

	.retro-subtitle {
		font-size: 20px;
		color: var(--text-dim);
		margin-top: 6px;
	}

	.retro-panel {
		border: 1px solid var(--panel-border);
		border-radius: 14px;
		padding: 20px;
		background: var(--panel-bg);
		box-shadow: 0 0 18px rgba(0, 255, 209, 0.15);
	}

	.retro-panel h3 {
		font-family: 'Press Start 2P', cursive;
		font-size: 16px;
		margin-bottom: 12px;
		color: var(--neon-cyan);
	}

	.stButton > button {
		font-family: 'Press Start 2P', cursive;
		font-size: 14px;
		color: #05000d;
		background: linear-gradient(135deg, var(--neon-cyan), var(--neon-pink));
		border: none;
		border-radius: 10px;
		padding: 12px 18px;
		box-shadow: 0 0 12px rgba(0, 255, 209, 0.5);
		transition: transform 0.2s ease, box-shadow 0.2s ease;
	}

	.stButton > button:hover {
		transform: translateY(-2px) scale(1.02);
		box-shadow: 0 0 18px rgba(255, 79, 216, 0.7);
	}

	.stTextInput > div > div > input,
	.stSelectbox > div > div > div {
		background: rgba(6, 10, 20, 0.85);
		border: 1px solid rgba(0, 255, 209, 0.4);
		color: var(--text-main);
	}

	.stTextArea > div > textarea {
		background: rgba(6, 10, 20, 0.85);
		border: 1px solid rgba(255, 79, 216, 0.4);
		color: var(--text-main);
	}

	.retro-output {
		white-space: pre-wrap;
		font-size: 20px;
		line-height: 1.45;
		color: var(--text-main);
	}

	.retro-chip {
		display: inline-block;
		padding: 6px 12px;
		border-radius: 999px;
		background: rgba(125, 92, 255, 0.2);
		border: 1px solid rgba(125, 92, 255, 0.5);
		color: var(--neon-purple);
		margin-right: 8px;
		font-size: 16px;
	}

	.footer-note {
		font-size: 16px;
		color: var(--text-dim);
	}
	</style>
	"""


def _build_prompt(subject: str, topic: str, difficulty: str, extra: str) -> str:
	extras = f"\nAdditional constraints: {extra.strip()}" if extra.strip() else ""
	return textwrap.dedent(
		f"""
		### Instruction:
		Generate a multiple-choice question.

		Subject: {subject}
		Topic: {topic}
		Difficulty: {difficulty}
		{extras}

		### Response:
		"""
	).strip()


@st.cache_resource(show_spinner=False)
def _load_model(adapter_path: str):
	model_name = "mistralai/Mistral-7B-v0.1"
	device = "cuda" if torch.cuda.is_available() else "cpu"

	quantization_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16,
		bnb_4bit_use_double_quant=False,
	)

	tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
	base_model = AutoModelForCausalLM.from_pretrained(
		model_name,
		quantization_config=quantization_config,
		device_map="auto",
		trust_remote_code=True,
	)

	model = PeftModel.from_pretrained(base_model, adapter_path)
	model.eval()

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	return model, tokenizer, device, model_name


def _generate_quiz(model, tokenizer, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
	inputs = tokenizer(prompt, return_tensors="pt", padding=True)
	inputs = {k: v.to(model.device) for k, v in inputs.items()}

	with torch.no_grad():
		output_ids = model.generate(
			**inputs,
			max_new_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			do_sample=True,
			repetition_penalty=1.1,
			pad_token_id=tokenizer.eos_token_id,
			eos_token_id=tokenizer.eos_token_id,
		)

	response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
	return response


st.markdown(_css(), unsafe_allow_html=True)

st.markdown(
	"""
	<div class="retro-header">
		<div class="retro-title">QUIZ FORGE</div>
		<div class="retro-subtitle">Generate neon-bright quizzes from LoRA-powered model.</div>
	</div>
	""",
	unsafe_allow_html=True,
)

left, right = st.columns([0.44, 0.56], gap="large")

with left:
	st.markdown('<div class="retro-panel">', unsafe_allow_html=True)
	st.markdown("### Quiz Controls")
	subject = st.text_input("Subject", value="Machine Learning")
	topic = st.text_input("Topic", value="Optimizers")
	difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=0)
	extra = st.text_area("Extra constraints (optional)", placeholder="E.g. 4 options, include correct answer label")

	st.markdown("### Generation Settings")
	temperature = st.slider("Temperature", min_value=0.1, max_value=1.2, value=0.7, step=0.05)
	top_p = st.slider("Top-p", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
	max_tokens = st.slider("Max new tokens", min_value=64, max_value=256, value=200, step=8)

	adapter_path = st.text_input(
		"LoRA adapter path",
		value=os.path.join(os.getcwd(), "quiz-lora-adapter"),
		help="Point this at your adapter folder if you moved it.",
	)

	generate = st.button("Generate Quiz", use_container_width=True)
	st.markdown("</div>", unsafe_allow_html=True)

with right:
	st.markdown('<div class="retro-panel">', unsafe_allow_html=True)
	st.markdown("### Output Console")
	output_placeholder = st.empty()
	st.markdown("</div>", unsafe_allow_html=True)

if generate:
	if not os.path.isdir(adapter_path):
		output_placeholder.error("Adapter path not found. Check the folder path and try again.")
	else:
		prompt = _build_prompt(subject, topic, difficulty, extra)
		with st.spinner("Booting the neon engine..."):
			model, tokenizer, device, model_name = _load_model(adapter_path)
		with st.spinner("Synthesizing quiz..."):
			response = _generate_quiz(model, tokenizer, prompt, temperature, top_p, max_tokens)

		output_placeholder.markdown(
			"""
			<div>
				<span class="retro-chip">MODEL: {model}</span>
				<span class="retro-chip">DEVICE: {device}</span>
				<span class="retro-chip">ADAPTER: LOADED</span>
			</div>
			<div class="retro-output">{content}</div>
			""".format(model=model_name, device=device.upper(), content=response),
			unsafe_allow_html=True,
		)

st.markdown(
	"""
	<div class="footer-note">
		Tip: Keep prompts short and specific. The adapter will nudge the base model toward quiz style.
	</div>
	""",
	unsafe_allow_html=True,
)