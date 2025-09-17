import streamlit as st

def load_theme():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"

    theme = st.session_state["theme"]

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if theme == "light":
        local_css("style_light.css")
    else:
        local_css("style_dark.css")

    # Theme toggle button
    if st.button("🌗 Toggle Theme", key="theme_toggle", help="Switch between Light and Dark Mode"):
        st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"
        st.rerun()

    return theme

def show_title():
    st.markdown('<div class="centered-title">🛡 Vulnerability Hunters</div>', unsafe_allow_html=True)

def navigation_bar():
    pages = ["Home", "Analysis & Insights", "Custom Visualizations", "Timeline", "Recommendations"]
    cols = st.columns(len(pages))
    selected_page = None
    for i, page in enumerate(pages):
        if cols[i].button(page, key=f"nav_{i}"):
            selected_page = page
    if selected_page is None:
        selected_page = "Home"
    return selected_page
