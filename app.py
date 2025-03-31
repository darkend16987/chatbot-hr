# -*- coding: utf-8 -*-
import streamlit as st
import google.generativeai as genai
import json
import os

# --- Cấu hình ứng dụng ---
st.set_page_config(page_title="INNO HR Chatbot", page_icon="🤖")
st.title("🤖 Beta App Hỏi Đáp Nhân Sự INNO")
st.caption("Hỏi đáp dựa trên dữ liệu nội bộ (JSON)")

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

# 2. Cấu hình Google Generative AI
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

if not api_key:
    st.error("Lỗi: API Key chưa được cấu hình. Vui lòng đặt biến môi trường GEMINI_API_KEY hoặc cấu hình GOOGLE_API_KEY trong Streamlit Secrets.")
    st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
     st.error(f"Lỗi khi cấu hình Google AI với API Key: {e}")
     st.stop()

MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Hoặc 'gemini-pro' nếu cần
try:
    # Thêm cài đặt an toàn để chặn nội dung không phù hợp nếu cần
    # safety_settings = [
    #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    # ]
    model = genai.GenerativeModel(
        MODEL_NAME
        # , safety_settings=safety_settings # Bỏ comment nếu dùng safety_settings
        )
    st.sidebar.info(f"Sử dụng model: {MODEL_NAME}")
except Exception as e:
    st.error(f"Lỗi khi khởi tạo model '{MODEL_NAME}': {e}. Có thể model không tồn tại hoặc API key không hợp lệ/không có quyền truy cập.")
    st.stop()

# --- System Prompt ---
system_instruction_text = """
Bạn là trợ lý nhân sự của công ty INNO. Bạn chỉ được sử dụng dữ liệu dưới đây để trả lời câu hỏi.
Nếu thông tin không có trong dữ liệu, hãy trả lời: "Tôi không tìm thấy thông tin trong dữ liệu có sẵn."
Đừng đưa ra suy đoán hoặc thông tin không chắc chắn.
"""

# Cấu hình tạo nội dung
generation_config = genai.types.GenerationConfig(
    response_mime_type="text/plain",
    temperature=0.7 # Giữ nguyên nhiệt độ
)

# --- THAY ĐỔI QUAN TRỌNG: Hàm Helper để xử lý Stream ---
def stream_text_generator(stream):
    """
    Hàm này nhận stream từ API, lặp qua các chunk,
    và chỉ yield phần text của mỗi chunk.
    """
    for chunk in stream:
        # Kiểm tra xem chunk có dữ liệu text không trước khi yield
        # Một số chunk ban đầu hoặc cuối cùng có thể không có 'text'
        if hasattr(chunk, 'text') and chunk.text:
             yield chunk.text
        # Có thể thêm xử lý cho các phần khác của chunk nếu cần (ví dụ: lỗi, safety ratings)

# --- Phần Giao diện và Xử lý ---

# --- Giao diện nhập câu hỏi ---
st.write("💡 Nhập câu hỏi vào ô bên dưới và nhấn Enter để tìm kiếm thông tin nhân sự.")
user_question = st.text_input(
    "Nhập câu hỏi của bạn:",
    key="user_question",
    placeholder="Ví dụ: Cho tôi biết email của ông Đoàn Văn Động?"
)

response_placeholder = st.container()

if user_question:
    if knowledge_text:
        full_prompt = f"{system_instruction_text}\n\nDưới đây là dữ liệu nhân sự dạng JSON:\n```json\n{knowledge_text}\n```\n\nDựa vào dữ liệu trên, hãy trả lời câu hỏi sau:\n\"{user_question}\""
        contents = [full_prompt]

        try:
            response_placeholder.empty()
            with response_placeholder:
                 with st.spinner(f"Đang tìm kiếm câu trả lời với {MODEL_NAME}..."):
                    response_stream = model.generate_content(
                        contents,
                        stream=True,
                        generation_config=generation_config
                    )

                    st.markdown("**Câu trả lời:**")
                    # --- THAY ĐỔI QUAN TRỌNG: Sử dụng hàm helper ---
                    # Truyền stream gốc vào hàm helper,
                    # và truyền kết quả của hàm helper (chỉ chứa text) vào st.write_stream
                    st.write_stream(stream_text_generator(response_stream))

        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi gọi Gemini API: {e}")
            print(f"Error calling Gemini API: {e}")
    else:
         st.error("Lỗi: Không có dữ liệu kiến thức để xử lý câu hỏi.")

elif not user_question: # Chỉ hiển thị khi ô trống và không có lỗi gì khác
    pass # Placeholder đã hiển thị thông báo mặc định

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 - Beta App v0.3")