import os
import json
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai 
import streamlit as st  

# Setup
# Load environment API
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Load menu từ file CSV
menu_df = pd.read_csv("menu.csv", index_col=[0], encoding='utf-8')
# Load config ban đầu LLM
with open("config.json", "r", encoding='utf-8') as f:
    config = json.load(f)
    function = config.get("function", "giới thiệu nhà hàng")
    initial_bot_message = config.get("initial_bot_message", "Xin chào ! Ban cần hỗ trợ gì ")  

# Load nội dung LLM
model = genai.GenerativeModel("gemini-1.5-flash",
                              system_instruction=f"""
                              Bạn tên là PhoBot, một trợ lý AI có nhiệm vụ hỗ trợ giải đáp thông tin cho khách hàng đến nhà hàng Viet Cuisine.
                              Các chức năng mà bạn hỗ trợ gồm:
                              1. Giới thiệu nhà hàng Viet Cuisine: là một nhà hàng thành lập bởi người Việt, ở địa chỉ 329 Scottmouth, Georgia, USA
                              2. Giới thiệu menu của nhà hàng, gồm các món: {', '.join(menu_df['name'].to_list())}.
                              Ngoài hai chức năng trên, bạn không hỗ trợ chức năng nào khác. Đối với các câu hỏi ngoài chức năng mà bạn hỗ trợ, trả lời bằng 'Tôi đang không hỗ trợ chức năng này. Xin liên hệ nhân viên nhà hàng qua hotline 318-237-3870 để được trợ giúp.'
                              Hãy có thái độ thân thiện và lịch sự khi nói chuyện với khác hàng, vì khách hàng là thượng đế.
                              """
)

# Hàm trò chuyện của chatbot
def restaurant_chatbot():
    st.title("Restaurant Assistant Chatbot")
    st.write("Chào mừng bạn đến với PhoBot, trợ lý AI của nhà hàng Viet Cuisine! Bạn cân hỗ trợ gì?")
    st.write("Bạn có thể hỏi về nhà hàng, menu hoặc các thông tin khác liên quan đến nhà hàng.")
# Nếu chưa có lịch sử trò chuyện
    if 'conversation_log' not in st.session_state:
        st.session_state.conversation_log = [
            {"role": "system", "content": initial_bot_message}
        ]
# Nêu có lịch sử trò chuyện
    for message in st.session_state.conversation_log:
        if message['role'] == 'user':
            with st.chat_message(message['role']):
                st.write(message['content'])
    # Khi người dùng gửi prompt
    if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây:"):
        # Hiển thị prompt của người dùng
        with st.chat_message("user"):
            st.write(prompt)
        # Lưu prompt vào lịch sử trò chuyện log
        st.session_state.conversation_log.append({"role": "user", "content": prompt})
        # Gọi LLM để lấy phản hồi
        response = model.generate_content(prompt)
        bot_reply = response.text

        # Kiểm tra xem promt có đề cập đến menu hay không
        if "menu" in prompt.lower() or "món" in prompt.lower():
            bot_reply ='\n\n'.join([f"**{row['name']}**: {row['description']}" for idx, row in menu_df.iterrows()])
        else:
            response = model.generate_content(prompt)
            bot_reply = response.text
        # # Kiểm tra xem promt có đề cập đến giá hay không
        # if "menu" in prompt.lower() or "món" in prompt.lower():
        #     bot_reply ='\n\n'.join([f"**{row['name']}**: {row['description']}" for idx, row in menu_df.iterrows()])
        # else:
        #     response = model.generate_content(prompt)
        #     bot_reply = response.text
        # # Kiểm tra xem promt có đề cập đến nguyên liệu hay không
        # if "menu" in prompt.lower() or "món" in prompt.lower():
        #     bot_reply ='\n\n'.join([f"**{row['name']}**: {row['description']}" for idx, row in menu_df.iterrows()])
        # else:
        #     response = model.generate_content(prompt)
        #     bot_reply = response.text
        # Hiển thị câu trả lời từ LLM
        with st.chat_message("assistant"):
            st.write(bot_reply)
        # Lưu câu trả lời vào lịch sử trò chuyện log
        st.session_state.conversation_log.append({"role": "assistant", "content": bot_reply})

# Hàm chạy ứng dụng
if __name__ == "__main__":
    restaurant_chatbot()