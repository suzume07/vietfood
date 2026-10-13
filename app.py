
from pathlib import Path
from class_names import CLASS_NAMES
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"

st.set_page_config(
    page_title="VietFood68",
    page_icon="🍜",
    layout="wide"
)

st.title("VietFood68 - Nhận diện món ăn")
st.write("Tải ảnh lên để AI nhận diện các đối tượng trong ảnh.")

if not MODEL_PATH.exists():
    st.error(
        "Không tìm thấy best.pt. "
        "Hãy đặt file best.pt cùng thư mục với app.py."
    )
    st.stop()


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
except Exception as e:
    st.error(f"Không tải được model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Chọn ảnh món ăn",
    type=["jpg", "jpeg", "png", "webp"]
)

confidence = st.slider(
    "Ngưỡng độ tin cậy",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Ảnh gốc")
    st.image(image, use_container_width=True)

    if st.button("Nhận diện món ăn", type="primary"):
        with st.spinner("Đang nhận diện..."):
            try:
                result = model.predict(
                    source=image,
                    conf=confidence,
                    verbose=False
                )[0]

                annotated = result.plot()[:, :, ::-1]

                st.subheader("Kết quả nhận diện")
                st.image(
                    annotated,
                    channels="RGB",
                    use_container_width=True
                )

                if result.boxes is None or len(result.boxes) == 0:
                    st.warning("Chưa phát hiện được đối tượng nào.")
                else:
                    rows = []

                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        score = float(box.conf.item())

                        name = CLASS_NAMES.get(class_id, f"class_{class_id}")

                        rows.append({
                            "Tên đối tượng": name,
                            "Độ tin cậy (%)": round(score * 100, 2)
                        })

                    st.subheader("Danh sách đối tượng")
                    st.dataframe(
                        rows,
                        use_container_width=True,
                        hide_index=True
                    )
                    st.success(f"Phát hiện {len(rows)} đối tượng.")

            except Exception as e:
                st.error(f"Lỗi nhận diện: {e}")
