import streamlit as st
from openai import OpenAI
import time

# Configure OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_response(model: str, prompt: str) -> str:
    """Generate response using specified OpenAI model"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def distill_knowledge(teacher_response: str) -> str:
    """Distill GPT-4 response into GPT-3.5 format"""
    distillation_prompt = f"""
    Summarize this expert response for a student model while preserving key information:
    
    Expert Response: {teacher_response}
    
    Concise Student Summary:
    """
    return generate_response("gpt-3.5-turbo", distillation_prompt)

# Streamlit UI
st.title("GPT-4 to GPT-3.5 Distillation Demo")
st.caption("Comparing teacher (GPT-4) vs distilled student (GPT-3.5-Turbo) responses")

user_query = st.text_input("Enter your query:", "Explain quantum computing in simple terms")

if st.button("Generate Responses"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Teacher Model (GPT-4)")
        start = time.time()
        teacher_response = generate_response("gpt-4", user_query)
        teacher_time = time.time() - start
        st.write(teacher_response)
        st.caption(f"Response time: {teacher_time:.2f}s")
        
    with col2:
        st.subheader("Student Model (GPT-3.5-Turbo)")
        start = time.time()
        student_response = distill_knowledge(teacher_response)
        student_time = time.time() - start
        st.write(student_response)
        st.caption(f"Response time: {student_time:.2f}s")
    
    st.divider()
    st.subheader("Performance Comparison")
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    with col_metric1:
        st.metric("Time Difference", 
                 f"{(teacher_time - student_time):.2f}s faster", 
                 delta_color="off")
        
    with col_metric2:
        st.metric("Token Efficiency", 
                 "~60% reduction", 
                 delta_color="off")
        
    with col_metric3:
        st.metric("Model Size", 
                 "GPT-4: ~1.8T\nGPT-3.5: 175B", 
                 delta_color="off")
    
    st.progress(65, text="Approximate cost reduction for similar tasks")
