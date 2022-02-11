import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os
from slr_project_mirror.display import draw_styled_landmarks, mediapipe_detection, extract_keypoints_no_face, prob_viz
# pip install onnx

<<<<<<< HEAD
from onnx_tf.backend import prepare
=======
class TestOnnx():
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, "../models/slr_5.onnx")
        
        self.model = onnxruntime.InferenceSession(model_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.mp_holistic = mp.solutions.holistic
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
        
        self.mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
        
    def extract_keypoints_no_face(self, results):
        pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten(
        ) if results.pose_landmarks else np.zeros(33*2)
        #print("Length pose :",pose.shape)
        # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
        # ) if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*2)
        #print("Length lh :",lh.shape)
        rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*2)
        #print("Length rh :",rh.shape)
        #print("Total length :", np.concatenate([pose, lh, rh]).shape)
        return np.concatenate([pose, lh, rh])


    def mediapipe_detection(self, image, model):
        # COLOR CONVERSION BGR 2 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

>>>>>>> main

class TestOnnx():
    def __init__(self, model):
        
        self.model = model
        # Import the ONNX model to Tensorflow
        #self.model = prepare(self.model)
        
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
        print("Launching onnx model")

    def get_sign(self, sequence, actions) -> list:
        """
        Get sign from frames
        """
<<<<<<< HEAD
        ort_session = self.model
        out = ort_session.run(
            None,
            {'input.1': np.array([sequence]).astype(np.float32),}
        )
        #out = np.exp(out) / np.sum(np.exp(out))
        return (actions[np.argmax(out)], float(np.max(out)))
=======
        #data = self.extract_keypoints_no_face(results)
        ort_inputs = {model.get_inputs()[0].name: np.array([sequence], dtype=np.float32)}
        out = model.run(None, ort_inputs)[-1]
        # sprint(out.shape)
        # print(np.shape(out))
        out = np.exp(out) / np.sum(np.exp(out))
        
        return (actions[np.argmax(out)], float(np.max(out))) 
        # return le sign et la probability
>>>>>>> main

    def launch_test(self, actions, targeted_action,cap, RESOLUTION_X, RESOLUTION_Y): 

        """
        actions l'ensemble des actions, action l'action à réaliser, cap l'instance de IntelCamera
        """
        sequence = []
        sentence = []
        threshold = 0.9

        #cap = cv2.VideoCapture(0)
        count_valid = 0
        # Set mediapipe model
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:

                #RECUPERATION DES COORDONNEES
                # Read feed
                frame, depth = cap.next_frame()
                #frame= cv2.resize(frame,(RESOLUTION_X,RESOLUTION_Y))

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)
                image = cv2.resize(image,(RESOLUTION_Y,RESOLUTION_X))
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                #image = cv2.flip(image, 1)
                window = 0.5
                min_width, max_width = int((0.5-window/2)*RESOLUTION_Y), int((0.5+window/2)*RESOLUTION_Y)
                
                image = image[:, min_width:max_width]  
                

                #TEST DU MODELE
            
                # Creation d'une séquence de frames     
                keypoints = extract_keypoints_no_face(results)       
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
<<<<<<< HEAD
                    sign, probability = self.get_sign(sequence, actions)
=======
                    sign, probability = self.get_sign(self.model, sequence, actions)
                    
>>>>>>> main
                    # 3. Viz logic
                    if probability > threshold:
                        if len(sentence) > 0:
                            if sign != sentence[-1]:
                                sentence.append(sign)
                        else:
                            sentence.append(sign)

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(sign, probability,actions, image, self.colors, targeted_action)
                    if(sign == targeted_action):
                        count_valid +=1
                    else : count_valid = 0

                    if(count_valid ==10):
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
            #cap.release()