import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./notebooks/mask_model.keras')

class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']
labels_dict = {0: "weared_incorrect", 1: 'with_mask', 2: 'without_mask'}
colors_dict = {0: (0, 255, 255), 1:(0, 255, 0), 2: (0, 0, 255)}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        resized = cv2.resize(face_img, (96, 96))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        reshaped = np.expand_dims(resized, axis=0)
        
        result = model.predict(reshaped, verbose=0)
        label_index = np.argmax(result)
        confidence = np.max(result)
        
        if confidence > 0.50:
            label = labels_dict[label_index]
            color = colors_dict[label_index]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({int(confidence*100)}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    cv2.imshow('Mask Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()