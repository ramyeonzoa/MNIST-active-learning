import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go
from sklearn.cluster import MiniBatchKMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import cv2
import time

st.set_page_config(page_title="Interactive Semi-Supervised Learning", layout="wide")

st.title("🤖 Interactive Semi-Supervised Learning")
st.markdown("**MNIST 손글씨** - 당신의 피드백으로 AI가 1-2초 안에 똑똑해집니다!")

# 세션 상태 초기화
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
    st.session_state.accuracy_history = []
    st.session_state.labeled_count = 0

@st.cache_data
def load_mnist_fast():
    """빠른 MNIST 로드"""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data[:10000] / 255.0, mnist.target[:10000].astype(int)  # 10k만 사용
    return X, y

def initialize_model():
    """모델 초기화 - 빠르게!"""
    with st.spinner("모델 초기화 중... (10초)"):
        # 데이터 로드
        X, y = load_mnist_fast()
        
        # 각 클래스별로 5개씩만 라벨링
        labels = np.full(len(X), -1)
        for digit in range(10):
            digit_indices = np.where(y == digit)[0][:50]
            selected = np.random.choice(digit_indices, size=5, replace=False)
            labels[selected] = digit
        
        # 빠른 KMeans (30개 클러스터)
        kmeans = MiniBatchKMeans(n_clusters=30, batch_size=500, random_state=42)
        kmeans.fit(X)
        
        # Label Propagation
        label_prop = LabelPropagation(kernel='knn', n_neighbors=15, max_iter=50)
        label_prop.fit(X, labels)
        
        # 테스트 세트
        X_test, y_test = X[8000:], y[8000:]
        
        # 세션에 저장
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.labels = labels
        st.session_state.label_prop = label_prop
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.model_initialized = True
        st.session_state.labeled_count = 50
        
        # 초기 정확도
        initial_acc = accuracy_score(y_test, label_prop.predict(X_test))
        st.session_state.accuracy_history = [initial_acc]

def preprocess_canvas(canvas_image):
    """캔버스 이미지를 MNIST 형식으로 변환"""
    if canvas_image is None:
        return None
    
    # RGB to Grayscale
    gray = cv2.cvtColor(canvas_image[:, :, :3], cv2.COLOR_RGB2GRAY)
    
    # 28x28로 리사이즈
    resized = cv2.resize(gray, (28, 28))
    
    # 정규화
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized.reshape(1, -1)

def quick_update(new_sample, true_label):
    """1-2초 안에 업데이트"""
    start_time = time.time()
    
    # 가장 가까운 50개 샘플 찾기
    distances = np.sum((st.session_state.X - new_sample) ** 2, axis=1)
    nearest_50 = np.argsort(distances)[:50]
    
    # 이 중에서 unlabeled인 것들만 업데이트
    for idx in nearest_50:
        if st.session_state.labels[idx] == -1:
            st.session_state.labels[idx] = true_label
    
    # 빠른 재학습 (일부만)
    labeled_indices = np.where(st.session_state.labels != -1)[0]
    if len(labeled_indices) > 100:
        sample_indices = np.random.choice(labeled_indices, 100, replace=False)
        st.session_state.label_prop.fit(
            st.session_state.X[sample_indices], 
            st.session_state.labels[sample_indices]
        )
    
    # 새 정확도 계산
    new_acc = accuracy_score(
        st.session_state.y_test, 
        st.session_state.label_prop.predict(st.session_state.X_test)
    )
    st.session_state.accuracy_history.append(new_acc)
    st.session_state.labeled_count = np.sum(st.session_state.labels != -1)
    
    elapsed = time.time() - start_time
    return new_acc, elapsed

# 모델 초기화 버튼
if not st.session_state.model_initialized:
    if st.button("🚀 모델 초기화 (10초)", type="primary"):
        initialize_model()
        st.success("모델 초기화 완료!")
        st.rerun()
else:
    # 메인 UI
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("✏️ 숫자 그리기")
        
        canvas_result = st_canvas(
            stroke_width=15,
            stroke_color="black",
            background_color="white",
            height=200,
            width=200,
            drawing_mode="freedraw",
            key="canvas"
        )
        
        if st.button("🎯 예측하기"):
            if canvas_result.image_data is not None:
                image = preprocess_canvas(canvas_result.image_data)
                if image is not None:
                    pred = st.session_state.label_prop.predict(image)[0]
                    proba = st.session_state.label_prop.predict_proba(image)[0]
                    confidence = np.max(proba)
                    
                    st.session_state.current_pred = pred
                    st.session_state.current_conf = confidence
                    st.session_state.current_image = image
    
    with col2:
        st.subheader("🎯 결과")
        
        if 'current_pred' in st.session_state:
            st.metric("예측", f"{st.session_state.current_pred}", f"신뢰도: {st.session_state.current_conf:.1%}")
            
            if st.session_state.current_conf < 0.8:
                st.warning("⚠️ 확신 부족!")
                
                correct = st.selectbox("정답:", range(10), index=int(st.session_state.current_pred))
                
                if st.button("⚡ 1초 학습!", type="primary"):
                    old_acc = st.session_state.accuracy_history[-1]
                    new_acc, elapsed = quick_update(st.session_state.current_image, correct)
                    
                    improvement = (new_acc - old_acc) * 100
                    
                    st.success(f"✅ {elapsed:.1f}초 만에 학습!")
                    st.metric("정확도 변화", f"{new_acc:.1%}", f"+{improvement:.2f}%p")
                    st.balloons()
            else:
                st.success("✅ 확신 있는 예측!")
    
    with col3:
        st.subheader("📊 성능")
        
        # 현재 통계
        current_acc = st.session_state.accuracy_history[-1]
        st.metric("현재 정확도", f"{current_acc:.1%}")
        st.metric("라벨링된 샘플", f"{st.session_state.labeled_count}")
        
        # 정확도 변화 그래프
        if len(st.session_state.accuracy_history) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.accuracy_history,
                mode='lines+markers',
                name='정확도'
            ))
            fig.update_layout(
                title="실시간 성능 향상",
                xaxis_title="업데이트 횟수",
                yaxis_title="정확도",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 리셋 버튼
    if st.sidebar.button("🔄 처음부터 다시"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()