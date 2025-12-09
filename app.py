import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io

def smart_resize(img_pil, target_width, target_height):
    """顔認識をしてリサイズする関数（クラウド完全対応版）"""
    # Streamlitでアップされた画像をOpenCV形式に変換
    img_np = np.array(img_pil)
    # 色の並び順をRGBからBGRに変換（OpenCVの仕様）
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_cv.shape[:2]

    # ★ここを修正しました★
    # wgetでダウンロードせず、ライブラリ内のデータを直接使います
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    except Exception as e:
        # 万が一エラーが出たら顔認識なしで進める（空リストにする）
        faces = []

    # 中心の決定（デフォルトは画像のど真ん中）
    center_x, center_y = orig_w / 2, orig_h / 2
    
    if len(faces) > 0:
        # 顔が見つかったら、全ての顔を含む範囲の中心を計算
        min_x = np.min(faces[:, 0])
        min_y = np.min(faces[:, 1])
        max_x = np.max(faces[:, 0] + faces[:, 2])
        max_y = np.max(faces[:, 1] + faces[:, 3])
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

    # リサイズ倍率の計算（隙間なく埋めるCover戦略）
    scale = max(target_width / orig_w, target_height / orig_h)
    resized_w, resized_h = int(orig_w * scale), int(orig_h * scale)
    
    # 画像のリサイズ
    img_resized = img_pil.resize((resized_w, resized_h), Image.LANCZOS)
    
    # クロップ位置計算
    center_x_scaled = center_x * scale
    center_y_scaled = center_y * scale
    left = center_x_scaled - (target_width / 2)
    top = center_y_scaled - (target_height / 2)
    
    # はみ出し補正
    left = max(0, min(left, resized_w - target_width))
    top = max(0, min(top, resized_h - target_height))
    
    # クロップ実行
    final_img = img_resized.crop((left, top, left + target_width, top + target_height))
    return final_img

# --- アプリの画面構成 ---
st.set_page_config(page_title="簡単リサイズ", layout="wide")
st.title("📷 AI自動リサイズアプリ")
st.markdown("画像をアップロードするだけで、人物を中心に自動トリミングします。")

# ファイルアップロード
uploaded_file = st.file_uploader("ここに画像をドラッグ＆ドロップ", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 画像を表示
    image = Image.open(uploaded_file)
    st.image(image, caption="元の画像", width=400)
    st.divider()
    
    st.subheader("👇 変換結果")
    
    # 作りたいサイズのリスト
    targets = [
        (1080, 1080, "正方形 (1:1)"),
        (1920, 1080, "横長 (16:9)"),
        (600, 400, "バナー (3:2)")
    ]

    # 3列に並べる
    cols = st.columns(3)
    
    for i, (w, h, label) in enumerate(targets):
        # リサイズ処理実行
        resized_img = smart_resize(image, w, h)
        
        # 画面に表示
        with cols[i]:
            st.write(f"**{label}** ({w}x{h})")
            st.image(resized_img, use_container_width=True)
            
            # ダウンロードボタン作成
            buf = io.BytesIO()
            resized_img.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()
            
            st.download_button(
                label=f"📥 保存 ({w}x{h})",
                data=byte_im,
                file_name=f"resized_{w}x{h}.jpg",
                mime="image/jpeg"
            )
