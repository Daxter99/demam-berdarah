import streamlit as st
import time

# Fungsi untuk hitungan mundur
def countdown_timer(seconds):
    countdown_placeholder = st.empty()  # Placeholder untuk memperbarui tulisan
    for i in range(seconds, 0, -1):
        countdown_placeholder.markdown(f"### Halaman akan direfresh dalam **{i} detik**...")
        time.sleep(1)
    countdown_placeholder.markdown("### Menyegarkan halaman...")
    st.experimental_rerun()  # Memuat ulang halaman

st.title("Contoh Hitungan Mundur di Streamlit")

# Tombol untuk memulai hitungan mundur
if st.button("Refresh"):
    countdown_timer(20)  # Hitungan mundur selama 20 detik