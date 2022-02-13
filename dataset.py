import cv2
import os
import mediapipe as mp
import numpy as np
from slr_project_mirror.display import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from slr_project_mirror.video import IntelVideoReader



# dans la création du dataset, je garde les points du visage au cas où


class CustomImageDataset():
    # class CustomImageDataset(Dataset):
    def __init__(self, actionsToAdd, nb_sequences, sequence_length, DATA_PATH_TRAIN, DATA_PATH_VALID, DATA_PATH_TEST, RESOLUTION_X,RESOLUTION_Y):
        self.actionsToAdd = actionsToAdd
        self.nb_sequences = nb_sequences
        self.sequence_length = sequence_length
        self.DATA_PATH_TRAIN =DATA_PATH_TRAIN
        self.DATA_PATH_VALID =DATA_PATH_VALID
        self.DATA_PATH_TEST =DATA_PATH_TEST
        self.RESOLUTION_X = RESOLUTION_X
        self.RESOLUTION_Y = RESOLUTION_Y
        self.cap = IntelVideoReader()
        print('dataset init')

    def __len__(self):
        
        return len(self.actionsToAdd)*len(self.nb_sequences)

    def __getitem__(self):


        for action in self.actionsToAdd:
            for sequence in range(self.nb_sequences):
                try:
                    if(sequence<self.nb_sequences*80/100):
                        os.makedirs(os.path.join(
                        self.DATA_PATH_TRAIN, action, str(sequence)))
                    elif(self.nb_sequences*80/100 <= sequence and sequence < self.nb_sequences*90/100):
                        os.makedirs(os.path.join(
                        self.DATA_PATH_VALID, action, str(int(sequence-self.nb_sequences*80/100))))
                    else:
                        os.makedirs(os.path.join(
                        self.DATA_PATH_TEST, action, str(int(sequence-self.nb_sequences*90/100))))
                    print("sucess")
                   
                except:
                    pass
        
        # Set mediapipe model
        mp_holistic = mp.solutions.holistic  # Holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            # Loop through actionsToAdd
            for action in self.actionsToAdd:
                # Loop through sequences aka videos
                for sequence in range(self.nb_sequences):
                    # Loop through video length aka sequence length

                    if(sequence<self.nb_sequences*80/100):
                        DATA_PATH = self.DATA_PATH_TRAIN
                    elif(self.nb_sequences*80/100 <= sequence and sequence < self.nb_sequences*90/100):
                        DATA_PATH = self.DATA_PATH_VALID
                    else:
                        DATA_PATH = self.DATA_PATH_TEST

                    for frame_num in range(self.sequence_length):

                        # Read feed
                        frame, depth = self.cap.next_frame()

                        # Make detections
                        frame= cv2.resize(frame,(self.RESOLUTION_Y,self.RESOLUTION_X))

                        image, results = mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        draw_styled_landmarks(image, results)

                        window = 0.5
                        min_width, max_width = int((0.5-window/2)*self.RESOLUTION_Y), int((0.5+window/2)*self.RESOLUTION_Y)
                        
                        image = image[:, min_width:max_width]  
                        image = cv2.flip(image, 1)
                        # NEW Apply wait logic
                        if frame_num == 0:
                            cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {}'.format(action), (15, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.putText(image, 'Video Number {}'.format(sequence), (15, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else:
                            cv2.putText(image, 'Collecting frames for {}'.format(action), (15, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.putText(image, 'Video Number {}'.format(sequence), (15, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                                        
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)

                        # NEW Export keypoints
                        keypoints = extract_keypoints(results)
                        #keypoints represente toutes les donnees d'une frame
                        
                        if(sequence<self.nb_sequences*80/100):
                            npy_path = os.path.join(
                            DATA_PATH, action, str(sequence), str(frame_num))
                        elif(self.nb_sequences*80/100 <= sequence and sequence < self.nb_sequences*90/100):
                            npy_path = os.path.join(
                            DATA_PATH, action, str(int(sequence-self.nb_sequences*80/100)), str(frame_num))
                        else:
                            npy_path = os.path.join(
                            DATA_PATH, action, str(int(sequence-self.nb_sequences*90/100)), str(frame_num))
                        
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break    
                            
            cv2.destroyAllWindows()

        
            
