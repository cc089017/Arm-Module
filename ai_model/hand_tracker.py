import mediapipe as mp

class HandTracker:
    def __init__(self,
                 max_num_hands=2,
                 detection_conf=0.5,
                 tracking_conf=0.5,
                 model_complexity=1,
                 static_image_mode=False):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,       # ✨ 정밀도 UP
            min_detection_confidence=detection_conf, # ✨ 검출 기준 완화
            min_tracking_confidence=tracking_conf    # ✨ 추적 기준 완화
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.fingertip_indices = [4, 8, 12, 16, 20]

    def process(self, img_rgb):
        # (선택) 밝기·대비 보정
        # img_rgb = cv2.convertScaleAbs(img_rgb, alpha=1.3, beta=10)
        return self.hands.process(img_rgb)

