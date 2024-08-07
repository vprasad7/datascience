import streamlit as st
from streamlit_float import *
import generate_response as resp

float_init(theme=True, include_unstable_primary=False)

def chat_content():
    st.session_state['contents'].append(st.session_state.content)

def main():
    ##st.set_page_config(layout="wide") 
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True) #Reducing white space at the top

    if 'contents' not in st.session_state:
        st.session_state['contents'] = []
        border = False
    else:
        border = True

    # Setting page title and heading
    st.title("CoP Chatbot")
    st.markdown('#')
    #st.header("Chat Agent to Answer question on visulaization tools!!")
    #st.header("CoP Chatbot")
    col1,col3,col2= st.columns([7,0.5,5],gap="small")


    col1, col2 = st.columns([1,2])
    with col1:
        with st.container(border=True):
            st.write('Hello Mate!! Please ask your question.')

    with col2:
        with st.container(border=border):
            history = st.container(height=400)
            with st.container():
                st.chat_input(key='content', on_submit=chat_content) 
                button_b_pos = "0rem"
                button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
                float_parent(css=button_css)

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if request := st.chat_input("Please ask your questions."):
                # Display user message in chat message container
                st.chat_message("user").markdown(request)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": request})
                llm_answer = resp.gen_resp(request)
                response = f"Output : {llm_answer}"
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
