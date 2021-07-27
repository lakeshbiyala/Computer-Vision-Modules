import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, static_image_mode=False, max_num_faces=5,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces,
                                                 self.min_detection_confidence,
                                                 self.min_tracking_confidence)  # initialize FaceMesh object
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:  # for each face
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                    face = []
                    for id, lm in enumerate(faceLms.landmark):
                        # print(lm)
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        #print(id, cx, cy)
                        face.append([cx, cy])
                faces.append(face)  # storing all faces

        return img, faces


def main():
    cap = cv2.VideoCapture("Videos/7.mp4")
    cTime = 0
    pTime = 0

    detector = FaceMeshDetector()

    while True:
        succcess, img = cap.read()
        img, faces = detector.findFaceMesh(img)

        if len(faces) != 0:
            print(len(faces))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
