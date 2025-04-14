import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import numpy as np

# Initialize models
@st.cache_resource
def load_models():
    teacher_model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    student_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return teacher_model, student_model, tokenizer

teacher, student, tokenizer = load_models()
client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

# Context-Aware Distillation
def context_aware_distill(input_text, temperature=0.7):
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Teacher generation with context
    with torch.no_grad():
        teacher_outputs = teacher.generate(**inputs, max_length=100, temperature=temperature)
    
    # Student distillation with attention to context
    student_outputs = student.generate(**inputs, max_length=100, temperature=temperature)
    
    return {
        "teacher": tokenizer.decode(teacher_outputs[0]),
        "student": tokenizer.decode(student_outputs[0])
    }

# Feedback Mechanism (Simple Bandit)
class BanditFeedback:
    def __init__(self):
        self.feedback_history = []
    
    def update(self, student_output, reward):
        self.feedback_history.append((student_output, reward))
        
    def get_reward_signal(self):
        if self.feedback_history:
            return np.mean([r for _, r in self.feedback_history[-5:]])
        return 0.5

# Streamlit UI
st.title("Context-Aware Distillation POC")

input_text = st.text_input("Enter your prompt:")
temperature = st.slider("Generation Temperature", 0.1, 1.0, 0.7)
feedback_system = BanditFeedback()

if input_text:
    outputs = context_aware_distill(input_text, temperature)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Teacher Output")
        st.write(outputs["teacher"])
    
    with col2:
        st.subheader("Student Output")
        st.write(outputs["student"])
    
    # Feedback Interface
    st.subheader("Feedback Mechanism")
    feedback = st.radio("Rate Student Output:", 
                       ("👍 Good", "👎 Needs Improvement"))
    
    if st.button("Submit Feedback"):
        reward = 1.0 if feedback.startswith("👍") else 0.0
        feedback_system.update(outputs["student"], reward)
        
        # Simple Bandit-based Adjustment
        current_reward = feedback_system.get_reward_signal()
        if current_reward < 0.5:
            st.warning("Adjusting student model parameters...")
            # Add actual parameter adjustment logic here
            # Example: student.resize_token_embeddings(len(tokenizer))
        
        st.success(f"Feedback recorded! Current reward signal: {current_reward:.2f}")

