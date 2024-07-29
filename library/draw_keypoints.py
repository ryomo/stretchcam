import cv2

# Keypoints to be checked when stretching
KEYPOINTS_TO_BE_CHECKED = {
    # 0: "nose",
    # 1: "left_eye",
    # 2: "right_eye",
    # 3: "left_ear",
    # 4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    # 11: "left_hip",
    # 12: "right_hip",
    # 13: "left_knee",
    # 14: "right_knee",
    # 15: "left_ankle",
    # 16: "right_ankle",
}

# Index pairs that should be connected by a line
LINES = [
    (0, 1),
    (0, 2),
    (0, 5),
    (0, 6),
    (1, 3),
    (2, 4),
    (5, 11),
    (5, 6),
    (5, 7),
    (6, 12),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]


class DrawKeypoints:

    def __init__(self, keypoint_score_th) -> None:
        self.keypoint_score_th = keypoint_score_th

    def draw(
        self,
        image,
        elapsed_time,
        keypoints,
    ):

        # Draw Keypoints
        self._draw_keypoints(image, keypoints)

        # Draw Lines
        self._draw_lines(image, keypoints)

        # Elapsed Time
        self._draw_text(
            image,
            "Elapsed Time : " + "{:.1f}".format(elapsed_time * 1000) + "ms",
            (10, 30),
            0.7,
        )

        return image

    def draw_loaded_keypoints(self, image, loaded_keypoints):
        # Draw Lines
        image = self._draw_lines(
            image, loaded_keypoints, [(255, 0, 0), (0, 0, 255)], [10, 8]
        )

        # Draw Keypoints
        for i, _ in enumerate(loaded_keypoints):
            # Keypoints which are not in KEYPOINTS_TO_BE_CHECKED want not to be drawn, so set the score to 0
            if i not in KEYPOINTS_TO_BE_CHECKED:
                loaded_keypoints[i][2] = 0
        image = self._draw_keypoints(image, loaded_keypoints, [(255, 0, 0)], [50], [5])
        return image

    def are_keypoints_close(self, keypoints, loaded_keypoints, threshold=0.1):
        """ "
        Check if the distance between each point of keypoints and loaded_keypoints is less than the threshold
        """
        for keypoint_idx in KEYPOINTS_TO_BE_CHECKED:
            if (
                keypoints[keypoint_idx] is None
                or loaded_keypoints[keypoint_idx] is None
            ):
                continue

            x1, y1, _ = keypoints[keypoint_idx]
            x2, y2, _ = loaded_keypoints[keypoint_idx]

            # Euclidean distance
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

            if distance > threshold:
                return False

        return True

    def draw_completed_message(self, image, message):
        scale = 1.0
        thickness = 3

        # Center the text
        textsize = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[
            0
        ]
        x = (image.shape[1] - textsize[0]) // 2
        y = (image.shape[0] - textsize[1]) // 2

        self._draw_text(
            image,
            message,
            (x, y),
            scale,
            [(0, 0, 0), (255, 255, 255)],
            [thickness, thickness -1],
        )
        return image

    def _draw_keypoints(
        self,
        image,
        keypoints,
        colors: list = [(255, 255, 255), (0, 0, 0)],
        radiuses: list = [6, 3],
        thicknesses: list = [-1, -1],
    ):
        image_width, image_height = image.shape[1], image.shape[0]

        for i, keypoint in enumerate(keypoints):
            x, y, score = (
                int(keypoint[0] * image_width),
                int(keypoint[1] * image_height),
                keypoint[2],
            )

            # Check the score
            if score < self.keypoint_score_th:
                continue

            for i, _ in enumerate(colors):
                # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
                cv2.circle(image, (x, y), radiuses[i], colors[i], thicknesses[i])

        return image

    def _draw_lines(
        self,
        image,
        keypoints,
        line_colors: list = [(255, 255, 255), (0, 0, 0)],
        line_thicknesses: list = [4, 2],
    ):
        for line in LINES:
            idx1, idx2 = line

            if keypoints[idx1] is None or keypoints[idx2] is None:
                continue

            x1, y1, score1 = (
                int(keypoints[idx1][0] * image.shape[1]),
                int(keypoints[idx1][1] * image.shape[0]),
                keypoints[idx1][2],
            )
            x2, y2, score2 = (
                int(keypoints[idx2][0] * image.shape[1]),
                int(keypoints[idx2][1] * image.shape[0]),
                keypoints[idx2][2],
            )

            if score1 < self.keypoint_score_th or score2 < self.keypoint_score_th:
                continue

            for i, _ in enumerate(line_colors):
                # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
                cv2.line(image, (x1, y1), (x2, y2), line_colors[i], line_thicknesses[i])

        return image

    def _draw_text(
        self,
        image,
        text: str,
        point: tuple,
        scale=0.5,
        colors: list = [(255, 255, 255), (0, 0, 0)],
        thicknesses: list = [3, 2],
    ):
        for i, _ in enumerate(colors):
            # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
            cv2.putText(
                image,
                text,
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                colors[i],
                thicknesses[i],
                cv2.LINE_AA,
            )
