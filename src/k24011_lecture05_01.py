import numpy as np
import cv2
from my_module.K21999.lecture05_camera_image_capture import MyVideoCapture

def k24011_lecture05_01():

    # カメラキャプチャ実行
    app = MyVideoCapture()
    app.run()

    # キャプチャ画像を取得
    capture_img: cv2.Mat = app.get_img()
    if capture_img is None:
        raise ValueError("カメラ画像が取得できませんでした。")

    # 置換対象（Google検索画面画像）を読み込み
    google_img: cv2.Mat = cv2.imread('images/google.png')
    if google_img is None:
        raise FileNotFoundError("google.png が見つかりません。")

    # サイズ取得
    g_height, g_width, _ = google_img.shape
    c_height, c_width, _ = capture_img.shape
    print("Google画像:", google_img.shape)
    print("キャプチャ画像:", capture_img.shape)

    # --- カメラ画像をタイル状に並べて背景を作成 ---
    tile_rows = 2   # 縦方向の繰り返し
    tile_cols = 2   # 横方向の繰り返し
    new_img = np.zeros((c_height * tile_rows, c_width * tile_cols, 3), dtype=np.uint8)

    for y in range(tile_rows):
        for x in range(tile_cols):
            new_img[c_height*y:c_height*(y+1), c_width*x:c_width*(x+1)] = capture_img[:, :]

    # タイル背景をGoogle画像サイズにリサイズ
    tiled_capture = cv2.resize(new_img, (g_width, g_height))

    # --- 白色部分を置換 ---
    result_img = np.copy(google_img)
    for y in range(g_height):
        for x in range(g_width):
            b, g, r = google_img[y, x]
            if (b, g, r) == (255, 255, 255):
                result_img[y, x] = tiled_capture[y, x]

    # --- 保存と表示 ---
    output_path = 'output_images/lecture05_k24011.png'
    cv2.imwrite(output_path, result_img)
    print(f"画像を保存しました: {output_path}")

    cv2.imshow('result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()