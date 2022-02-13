import torch
import cv2
import numpy as np
import mediapipe as mp
from slr_project_mirror.display import draw_styled_landmarks, mediapipe_detection, extract_keypoints_no_face, prob_viz


class Test():
    def __init__(self, model):
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

        self.model = model
        print("Launching pth model")
        
    def launch_test(self, actions, targeted_action, cap, RESOLUTION_X, RESOLUTION_Y):
        count_valid = 0
        threshold = 0.9
        mp_holistic = mp.solutions.holistic
        sequence = []
        sentence = []
        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:

                # Read feed
                frame, depth = cap.next_frame()
                #frame= cv2.resize(frame,(RESOLUTION_X,RESOLUTION_Y))

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)
                image = cv2.resize(image, (RESOLUTION_Y, RESOLUTION_X))

                # Draw landmarks
                draw_styled_landmarks(image, results)
                #image = cv2.flip(image, 1)
                window = 0.5
                min_width, max_width = int(
                    (0.5-window/2)*RESOLUTION_Y), int((0.5+window/2)*RESOLUTION_Y)

                image = image[:, min_width:max_width]

                # 2. Prediction logic
                keypoints = extract_keypoints_no_face(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    #res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    res = torch.softmax(
                        self.model(torch.tensor(
                            sequence, dtype=torch.float).cuda().unsqueeze(0)),
                        dim=1
                    ).cpu().detach().numpy()[0]

                    sign = actions[np.argmax(res)]
                    probability = res[np.argmax(res)]
                    print(sign)

                    # 3. Viz logic
                    if np.max(res) > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(sign, probability, image, self.colors, targeted_action)
                    if(sign == targeted_action):
                        count_valid += 1
                    else:
                        count_valid = 0

                    if(count_valid == 10):
                        print("VALIDATED")
                        break

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('My window', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            # cap.release()
