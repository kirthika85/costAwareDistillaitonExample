import asyncio
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from streamlit.web import cli as st_cli
import sys

# Configure to prevent file watcher error
sys.argv = ["streamlit", "run", sys.argv[0]]
st.set_option('server.fileWatcherType', 'none')

# Model loading with error handling
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    try:
        teacher = AutoModelForCausalLM.from_pretrained("gpt2-large")
        student = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return teacher, student, tokenizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

teacher, student, tokenizer = load_models()

# Context-aware distillation function
def context_aware_distill(input_text, temperature=0.7):
    try:
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Teacher generation with context
        with torch.no_grad():
            teacher_out = teacher.generate(
                **inputs,
                max_length=100,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Student distillation
        student_out = student.generate(
            **inputs,
            max_length=100,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return {
            "teacher": tokenizer.decode(teacher_out[0], skip_special_tokens=True),
            "student": tokenizer.decode(student_out[0], skip_special_tokens=True)
        }
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None

# Feedback system with Bandit
class BanditFeedback:
    def __init__(self):
        self.feedback_history = []
    
    def update(self, output, reward):
        self.feedback_history.append((output, reward))
    
    def get_reward(self):
        return np.mean([r for _, r in self.feedback_history]) if self.feedback_history else 0.5

# Streamlit UI
def main():
    st.title("🛠️ Context-Aware Distillation POC")
    
    feedback_system = BanditFeedback()
    
    with st.form("distillation_form"):
        input_text = st.text_input("Enter your prompt:", "Explain quantum computing in simple terms")
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
        submitted = st.form_submit_button("Generate")
    
    if submitted and input_text:
        with st.spinner("Distilling knowledge..."):
            results = context_aware_distill(input_text, temperature)
        
        if results:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🧠 Teacher Model")
                st.write(results["teacher"])
            
            with col2:
                st.subheader("🎓 Student Model")
                st.write(results["student"])
            
            # Feedback interface
            st.subheader("📝 Feedback System")
            feedback = st.radio("Rate student output:", 
                              ("👍 Good", "👎 Needs Improvement"),
                              horizontal=True)
            
            if st.button("Submit Rating"):
                reward = 1.0 if feedback.startswith("👍") else 0.0
                feedback_system.update(results["student"], reward)
                
                current_reward = feedback_system.get_reward()
                st.success(f"Reward signal updated: {current_reward:.2f}/1.00")
                
                if current_reward < 0.5:
                    st.warning("Adjusting student model parameters...")
                    # Add actual adjustment logic here

# Entry point
if __name__ == "__main__":
    if st_cli._is_running_with_streamlit:
        main()
    else:
        sys.exit(st_cli.main())
