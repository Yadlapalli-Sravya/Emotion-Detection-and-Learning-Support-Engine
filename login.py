import streamlit as st
from auth import login_user, register_user

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

st.set_page_config(
    page_title="Login",
    page_icon="🔐",
    layout="centered"
)


# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
    background:linear-gradient(135deg,#eef2ff,#ffffff);
}



.title{
    text-align:center;
    font-size:40px;
    font-weight:700;
    color:#2563EB;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.stButton>button{
    width:100%;
    height:50px;
    background:linear-gradient(90deg,#2563EB,#4F46E5);
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- CARD ---------------- #

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown(
"""
<div class='title'>
🎓 Emotion Detection &
Learning Support Engine
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
Login to continue
</div>
""",
unsafe_allow_html=True
)

page = st.radio(
    "",
    ["Login","Register"],
    horizontal=True
)

# ---------------- LOGIN ---------------- #

if page=="Login":

    email=st.text_input("Email")

    password=st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        user=login_user(email,password)

        if user:
            st.session_state.logged_in = True
            st.session_state.user = user[1]
            st.success("Login Successful!")
            st.switch_page("pages/app.py")
        else:

            st.error("Invalid Email or Password")

# ---------------- REGISTER ---------------- #

else:

    name=st.text_input("Full Name")

    email=st.text_input("Email")

    password=st.text_input(
        "Password",
        type="password"
    )

    if st.button("Create Account"):

        success=register_user(
            name,
            email,
            password
        )

        if success:

            st.success("Registration Successful!")

        else:

            st.error("Email already exists.")

st.markdown("</div>", unsafe_allow_html=True)