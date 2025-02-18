Control System with Hand Gestures

   •	This project is a simple mini-project that attempts to perform specific actions by recognizing hand gestures.
   •	The project operates based on the 4 hand gestures shown in the image below: (Gesture_1, Gesture_2, Gesture_3, Gesture_4)
![image](https://github.com/user-attachments/assets/ba8c2624-6048-465d-93bc-c0eb9b797525)

   •	Gesture_1 performs the play/pause operation.
   •	Gesture_2 performs the volume up operation.
   •	Gesture_3 performs the volume down operation.
   •	Gesture_14 performs the spacebar press operation.
	The assignment of actions to hand gestures has been completed.

![image](https://github.com/user-attachments/assets/37e525d7-5880-4292-8069-b93b06769ebf)

I created the architecture of the code as shown in the diagram above.

   •	Examination of the hand gesture dataset and preprocessing.
   •	I created data augmentation and a CNN model. I added 4 convolutional layers and 4 max pooling layers, 
   then used flattening to convert the image into a one-dimensional vector. 
   I added 256 units with a dense layer and then applied dropout to prevent overfitting by deactivating some units. 
   Finally, I created the output layer with the last dense layer.
   •	I performed tasks such as compiling, training, and saving the model.
   •	And finally,I wrote the code that will trigger actions based on the user's hand gestures.

