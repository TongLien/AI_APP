import streamlit as st
from temp import translate_zh_to_en, translate_zh_to_vi, PIVOT_AVAILABLE

# ================== Cấu hình giao diện ==================
st.set_page_config(page_title="Trình dịch Trung -> Anh / Việt", page_icon="🌐", layout="centered")

st.title("🌐 Trình dịch tiếng Trung")
st.write("Nhập văn bản tiếng Trung, hệ thống sẽ dịch sang **English** và (nếu có model) sang **Tiếng Việt**.")

# ================== Input ==================
txt = st.text_area("Nhập văn bản tiếng Trung:", height=150)

if st.button("Dịch ngay"):
    if not txt.strip():
        st.warning("Vui lòng nhập văn bản trước khi dịch.")
    else:
        with st.spinner("⏳ Đang dịch..."):
            try:
                en = translate_zh_to_en(txt)
                st.success("✅ Dịch xong!")

                st.subheader("👉 English")
                st.write(en)

                if PIVOT_AVAILABLE:
                    vi = translate_zh_to_vi(txt)
                    st.subheader("👉 Vietnamese (pivot)")
                    st.write(vi)
                else:
                    st.info("⚠️ Model En->Vi chưa sẵn sàng, chỉ có thể dịch sang English.")

            except Exception as e:
                st.error(f"❌ Lỗi khi dịch: {e}")

# ================== Footer ==================
st.markdown("---")
st.caption("🚀 Built with HuggingFace Transformers + Streamlit")
