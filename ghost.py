import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def ghost_artifacts(image, offset_pixels, ghost_edge_ratio=0.1, ghost_alpha=0.3):
    """
    입력 이미지에서 엣지를 검출한 후, 전체 엣지 픽셀 중 ghost_edge_ratio (예: 0.1, 10%)에 해당하는
    연속된 영역을 선택합니다. 한 번 결정된 랜덤 방향으로 offset_pixels만큼 이동시킨 후,
    두 가지 방식의 고스트 효과를 생성합니다.
      1. Red Overlay 방식: 이동된 영역에 빨간색 오버레이 적용
      2. Blurred Ghost 방식: 원본 이미지를 동일 방향으로 이동시켜 GaussianBlur 처리 후 블렌딩
    """
    # 엣지 검출을 위해 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Canny 엣지 검출
    edges = cv2.Canny(gray, 100, 200)
    
    # 전체 엣지 픽셀 수 및 목표 픽셀 수 계산
    total_edge_pixels = np.count_nonzero(edges)
    target_pixels = total_edge_pixels * ghost_edge_ratio
    
    # 컨투어(연속된 엣지 영역) 찾기 및 리스트 변환
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    
    # 선택할 영역 마스크 생성 (전체 엣지의 ghost_edge_ratio만큼 누적)
    selected_mask = np.zeros_like(edges)
    current_pixels = 0
    np.random.shuffle(contours)
    for cnt in contours:
        temp_mask = np.zeros_like(edges)
        cv2.drawContours(temp_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        cnt_pixels = np.count_nonzero(temp_mask)
        if current_pixels == 0 or current_pixels + cnt_pixels <= target_pixels:
            selected_mask = cv2.bitwise_or(selected_mask, temp_mask)
            current_pixels += cnt_pixels
        if current_pixels >= target_pixels:
            break

    # 한 번 결정된 랜덤 방향 (0 ~ 2π)
    angle = np.random.uniform(0, 2 * np.pi)
    dx = int(round(offset_pixels * np.cos(angle)))
    dy = int(round(offset_pixels * np.sin(angle)))
    
    height, width = selected_mask.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_mask = cv2.warpAffine(selected_mask, M, (width, height))
    
    # 1. Red Overlay 방식 결과 생성
    if len(image.shape) == 2:
        composite_red = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        composite_red = image.copy()
    red_overlay = np.zeros_like(composite_red)
    red_overlay[shifted_mask == 255] = [0, 0, 255]  # BGR: 빨간색
    alpha_red = 0.5
    composite_red = cv2.addWeighted(composite_red, 1, red_overlay, alpha_red, 0)
    
    # 2. Blurred Ghost 방식 결과 생성
    if len(image.shape) == 2:
        composite_blur = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        composite_blur = image.copy()
    ghost_layer = cv2.warpAffine(image, M, (width, height))
    ghost_layer = cv2.GaussianBlur(ghost_layer, (3, 3), 0)
    mask_bool = shifted_mask.astype(bool)
    composite_blur[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - ghost_alpha,
                                                 ghost_layer[mask_bool], ghost_alpha, 0)
    
    return composite_red, composite_blur

# --- Tkinter GUI 구성 ---

# 전역 변수로 선택된 파일 경로 및 Tkinter 이미지 객체 보관
selected_file = ""
orig_photo = None
red_photo = None
blur_photo = None

def select_file():
    global selected_file, orig_photo
    filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    selected_file = filedialog.askopenfilename(title="이미지 파일 선택", filetypes=filetypes)
    if selected_file:
        file_label.config(text=os.path.basename(selected_file))
        # 파일 선택 시 원본 이미지 로드 및 좌측 패널에 표시
        image = cv2.imread(selected_file)
        if image is None:
            messagebox.showerror("에러", "이미지를 불러올 수 없습니다.")
            return
        # BGR -> RGB 변환 후 PIL 이미지로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        # 필요에 따라 사이즈 조정 (여기서는 300x300으로 고정)
        pil_image = pil_image.resize((300, 300))
        global orig_photo
        orig_photo = ImageTk.PhotoImage(pil_image)
        orig_label.config(image=orig_photo)
    else:
        file_label.config(text="선택된 파일이 없습니다.")

def process_image():
    global selected_file, red_photo, blur_photo
    if not selected_file:
        messagebox.showerror("에러", "먼저 이미지 파일을 선택하세요.")
        return
    try:
        offset_pixels = int(offset_entry.get())
        ghost_edge_ratio = float(edge_ratio_entry.get())
        ghost_alpha = float(ghost_alpha_entry.get())
    except ValueError:
        messagebox.showerror("에러", "입력값이 올바르지 않습니다. 숫자 값을 확인해 주세요.")
        return

    image = cv2.imread(selected_file)
    if image is None:
        messagebox.showerror("에러", "이미지를 불러올 수 없습니다.")
        return
    
    result_red, result_blur = ghost_artifacts(image, offset_pixels, ghost_edge_ratio, ghost_alpha)
    
    # 결과 이미지를 BGR -> RGB 변환 후 PIL 이미지로 변환, 사이즈 조정 (300x300)
    result_red_rgb = cv2.cvtColor(result_red, cv2.COLOR_BGR2RGB)
    pil_red = Image.fromarray(result_red_rgb).resize((300, 300))
    red_photo = ImageTk.PhotoImage(pil_red)
    red_label.config(image=red_photo)
    
    result_blur_rgb = cv2.cvtColor(result_blur, cv2.COLOR_BGR2RGB)
    pil_blur = Image.fromarray(result_blur_rgb).resize((300, 300))
    blur_photo = ImageTk.PhotoImage(pil_blur)
    blur_label.config(image=blur_photo)
    
    # 결과 이미지 저장 (자동 저장)
    base, ext = os.path.splitext(selected_file)
    red_save_path = base + "_ghost_red" + ext
    blur_save_path = base + "_ghost_blur" + ext
    cv2.imwrite(red_save_path, result_red)
    cv2.imwrite(blur_save_path, result_blur)
    messagebox.showinfo("저장 완료", f"결과 이미지가 자동 저장되었습니다.\n{os.path.basename(red_save_path)}\n{os.path.basename(blur_save_path)}")

# Tkinter 창 생성 및 레이아웃 구성
root = tk.Tk()
root.title("Ghost Artifact Generator")

# 상단: 파일 선택 및 파라미터 입력 영역
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

# 파일 선택 영역
file_frame = tk.Frame(top_frame)
file_frame.grid(row=0, column=0, padx=5)
select_btn = tk.Button(file_frame, text="파일 선택", command=select_file)
select_btn.pack(side=tk.LEFT, padx=5)
file_label = tk.Label(file_frame, text="선택된 파일이 없습니다.")
file_label.pack(side=tk.LEFT)

# 파라미터 입력 영역
param_frame = tk.Frame(top_frame)
param_frame.grid(row=0, column=1, padx=20)
tk.Label(param_frame, text="Offset Pixels:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
offset_entry = tk.Entry(param_frame, width=10)
offset_entry.grid(row=0, column=1, padx=5, pady=5)
offset_entry.insert(0, "10")  # 기본값

tk.Label(param_frame, text="엣지 비율 (0~1):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
edge_ratio_entry = tk.Entry(param_frame, width=10)
edge_ratio_entry.grid(row=1, column=1, padx=5, pady=5)
edge_ratio_entry.insert(0, "0.1")  # 기본값

tk.Label(param_frame, text="고스트 블러 비율 (0~1):").grid(row=2, column=0, sticky="e", padx=5, pady=5)
ghost_alpha_entry = tk.Entry(param_frame, width=10)
ghost_alpha_entry.grid(row=2, column=1, padx=5, pady=5)
ghost_alpha_entry.insert(0, "0.3")  # 기본값

process_btn = tk.Button(top_frame, text="Process", command=process_image)
process_btn.grid(row=0, column=2, padx=5)

# 하단: 이미지 결과 표시 영역 (3개 칼럼)
bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=10)

# 왼쪽: 원본 이미지
orig_frame = tk.Frame(bottom_frame, bd=2, relief="sunken")
orig_frame.grid(row=0, column=0, padx=5)
tk.Label(orig_frame, text="Original").pack()
orig_label = tk.Label(orig_frame)
orig_label.pack()

# 중간: 빨간색 오버레이 결과
red_frame = tk.Frame(bottom_frame, bd=2, relief="sunken")
red_frame.grid(row=0, column=1, padx=5)
tk.Label(red_frame, text="Ghost - Red Overlay").pack()
red_label = tk.Label(red_frame)
red_label.pack()

# 오른쪽: 블러 처리 결과
blur_frame = tk.Frame(bottom_frame, bd=2, relief="sunken")
blur_frame.grid(row=0, column=2, padx=5)
tk.Label(blur_frame, text="Ghost - Blurred").pack()
blur_label = tk.Label(blur_frame)
blur_label.pack()

root.mainloop()
