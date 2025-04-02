# -*- coding: utf-8 -*-

import streamlit as st
import google.generativeai as genai
import json
import os

# --- Cấu hình ứng dụng ---
st.set_page_config(page_title="INNO HR Chatbot", page_icon="🤖")
st.title("🤖 Trợ lý AI Hỏi Đáp Nhân Sự INNO")
st.caption("Hỏi đáp dựa trên dữ liệu nội bộ (JSON)")

# --- CSS tùy chỉnh để cố định ô nhập liệu ở cuối ---
# Lưu ý: CSS này dựa trên cấu trúc HTML hiện tại của Streamlit và có thể cần
# điều chỉnh nếu Streamlit cập nhật cấu trúc trong tương lai.
st.markdown("""
<style>
/* Target the container holding the text input */
/* Adjust selector if needed based on Streamlit version changes */
div.stTextInput {
    position: fixed; /* Cố định vị trí */
    bottom: 1rem; /* Khoảng cách từ đáy màn hình */
    left: 50%; /* Căn giữa theo chiều ngang */
    transform: translateX(-50%); /* Điều chỉnh căn giữa chính xác */
    width: calc(100% - 4rem); /* Chiều rộng, trừ đi padding hai bên */
    max-width: 736px; /* Giới hạn chiều rộng tối đa giống nội dung chính */
    padding: 0.75rem 1rem;
    background-color: #ffffff; /* Màu nền trắng (hoặc màu nền theme) */
    border-top: 1px solid #e6e6e6; /* Đường viền mờ phía trên */
    border-radius: 0.5rem; /* Bo góc nhẹ */
    box-shadow: 0 -2px 5px rgba(0,0,0,0.05); /* Đổ bóng nhẹ */
    z-index: 99; /* Đảm bảo nằm trên các thành phần khác */
}

/* Thêm padding vào cuối nội dung chính để không bị ô input che mất */
.main .block-container {
    padding-bottom: 6rem; /* Tăng padding để đủ chỗ cho ô input cố định */
}
</style>
""", unsafe_allow_html=True)

# --- Tải dữ liệu nhân sự từ file JSON ---
JSON_FILE_PATH = "data.json"

@st.cache_data
def load_knowledge(file_path):
    """Tải dữ liệu JSON mà không chiếm quá nhiều bộ nhớ"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)  # Trả về object JSON thay vì string
    except FileNotFoundError:
        st.sidebar.error(f"❌ Không tìm thấy file '{file_path}'!")
        return None
    except json.JSONDecodeError:
        st.sidebar.error(f"❌ File '{file_path}' không đúng định dạng JSON!")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải dữ liệu: {e}")
        return None

knowledge_text = load_knowledge(JSON_FILE_PATH)

if knowledge_text is None:
    st.error("Không thể khởi động ứng dụng do lỗi nghiêm trọng khi tải dữ liệu kiến thức. Vui lòng kiểm tra file data.json và thử lại.")
    st.stop()
else:
    st.sidebar.success(f"Đã tải thành công dữ liệu từ {JSON_FILE_PATH}")

# --- Cấu hình Google Generative AI ---
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

if not api_key:
    st.error("Lỗi: API Key chưa được cấu hình. Vui lòng đặt biến môi trường GEMINI_API_KEY hoặc cấu hình GOOGLE_API_KEY trong Streamlit Secrets.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Lỗi khi cấu hình Google AI với API Key: {e}")
    st.stop()

MODEL_NAME = "gemini-2.5-pro-exp-03-25"  # Hoặc 'gemini-pro' nếu cần

try:
    model = genai.GenerativeModel(MODEL_NAME)
    st.sidebar.info(f"Sử dụng model: {MODEL_NAME}")
except Exception as e:
    st.error(f"Lỗi khi khởi tạo model '{MODEL_NAME}': {e}. Có thể model không tồn tại hoặc API key không hợp lệ/không có quyền truy cập.")
    st.stop()

# --- System Prompt ---
system_instruction_text = """
Act as the HR information manager of INNO company. You have the skills and knowledge to read, remember, check, and respond to HR-related inquiries. When asked for information, you must only use the provided data to answer the question.

If the information is not available in the provided data, respond with: "I could not find this information in the available data."
Do not make guesses or provide uncertain information.

Your process for handling and responding to inquiries is as follows:

- Receive the question
- Analyze and understand the question and the user's concern in natural language
- (Important) Carefully check the information in the provided documents and extract all relevant and accurate details
- Respond in natural language
- Accept feedback if the response is incorrect.
"""

# --- Cấu hình tạo nội dung ---
generation_config = genai.types.GenerationConfig(
    response_mime_type="text/plain",
    temperature=0.7  # Giữ nguyên nhiệt độ
)

# --- Xử lý phản hồi dạng Stream ---
def stream_text_generator(stream):
    """Nhận stream từ API và chỉ xuất phần text"""
    for chunk in stream:
        if hasattr(chunk, 'text') and chunk.text:
            yield chunk.text

# --- Quản lý Lịch sử Chat (Session State) ---
# Khởi tạo danh sách tin nhắn nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hiển thị Lịch sử Chat ---
st.markdown("### Lịch sử trò chuyện")
# Tạo container để chứa lịch sử, cho phép cuộn nếu cần
# Không cần giới hạn chiều cao cố định nếu ô input đã cố định ở dưới
chat_container = st.container()

with chat_container:
    # Chỉ hiển thị tối đa 2 cặp hỏi đáp gần nhất (4 tin nhắn)
    history_limit = 4
    start_index = max(0, len(st.session_state.messages) - history_limit)
    for i in range(start_index, len(st.session_state.messages)):
        message = st.session_state.messages[i]
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Giao diện nhập câu hỏi (Sẽ được CSS di chuyển xuống dưới) ---
user_question = st.text_input(
    "Nhập câu hỏi của bạn:",
    key="user_question_input", # Key để quản lý input
    placeholder="Ví dụ: Cho tôi biết email của Nguyễn Văn A?",
    label_visibility="collapsed" # Ẩn label mặc định để tiết kiệm không gian
)

# --- Xử lý logic chính khi có câu hỏi mới ---
if user_question:
    # 1. Thêm câu hỏi của người dùng vào lịch sử và hiển thị ngay lập tức
    # Kiểm tra xem tin nhắn cuối cùng có phải là câu hỏi này không để tránh lặp lại khi rerun
    # ----- SỬA LỖI: TOÀN BỘ LOGIC BÊN DƯỚI PHẢI NẰM TRONG KHỐI IF NÀY -----
    if not st.session_state.messages or st.session_state.messages[-1].get("content") != user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Hiển thị câu hỏi mới nhất trong container lịch sử
        # ----- Thụt lề đúng -----
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_question)

        # ----- Thụt lề đúng cho toàn bộ khối xử lý prompt và API -----
        # 2. Xây dựng prompt
        # Chuyển JSON thành chuỗi (lưu ý: indent=2 sẽ tốn nhiều token hơn indent=None)
        knowledge_base_string = json.dumps(knowledge_text, ensure_ascii=False, indent=None) # <<< Sửa lại indent=None để tiết kiệm token
        full_prompt = f"{system_instruction_text}\n\nDưới đây là dữ liệu nhân sự:\n\n{knowledge_base_string}\n\nHãy trả lời câu hỏi sau:\n\"{user_question}\""

        # ----- THÊM LẠI DÒNG BỊ THIẾU VÀ THỤT LỀ ĐÚNG -----
        contents = [full_prompt]

        # 3. Gọi API và xử lý phản hồi
        try:
            # ----- Thụt lề đúng -----
            with chat_container:
                with st.spinner(f"🔍 Đang tìm kiếm câu trả lời với {MODEL_NAME}..."):
                    response_stream = model.generate_content(
                        contents, # <<< Sử dụng biến contents đã định nghĩa
                        stream=True,
                        generation_config=generation_config
                    )

                    # 4. Hiển thị câu trả lời dạng stream và lưu trữ
                    with st.chat_message("assistant"):
                        full_response = ""
                        text_stream_placeholder = st.empty()
                        text_generator = stream_text_generator(response_stream)
                        for chunk in text_generator:
                            full_response += chunk
                            text_stream_placeholder.markdown(full_response + "▌") # Con trỏ nhấp nháy
                        text_stream_placeholder.markdown(full_response) # Hiển thị đầy đủ

            # 5. Thêm câu trả lời đầy đủ vào lịch sử (sau khi stream xong)
            # ----- Thụt lề đúng -----
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Các dòng tùy chọn khác giữ nguyên thụt lề này
            # st.session_state.user_question_input = ""
            # st.rerun()

        except Exception as e:
             # ----- Thụt lề đúng -----
            st.error(f"🚨 Đã xảy ra lỗi khi gọi Gemini API: {e}")
            print(f"Error calling Gemini API: {e}")

# --- Không cần else cho if not st.session_state.messages... vì chỉ xử lý khi có câu hỏi MỚI ---

# --- Khối elif này để xử lý khi ô input trống ---
# elif not user_question: # Không cần thiết vì đã có xử lý hiển thị lịch sử ở trên
#     pass

# --- Thông tin thêm ---
# Có thể không cần thiết nữa nếu giao diện tập trung vào chat
# st.sidebar.markdown("---")
# st.sidebar.caption("© 2025 - Beta App v0.7") # Cập nhật version
