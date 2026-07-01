import tensorflow as tf
import os
import numpy as np
import cv2
import glob
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split




from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from mtcnn.mtcnn import MTCNN


# Check for GPU and set memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Using GPU: {gpus}. Mixed precision is DISABLED for stability.")
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("WARNING: No GPU found. Training stability prioritized over speed.")


# --- Albumentations Imports ---
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("FATAL ERROR: Albumentations not found. Please ensure 'albumentations' is installed.")
    exit()


# --- GLOBAL INITIALIZATION ---
try:
    face_detector = MTCNN()
except Exception as e:
    print(f"Error initializing MTCNN detector: {e}. Check MTCNN installation.")
    exit()


NUM_WORKERS = 1


##Configuration
IMG_SIZE = 224
BATCH_SIZE = 8  
EPOCHS = 20
FINE_TUNE_EPOCHS = 80
INITIAL_LR = 1e-4
FINE_TUNE_LR = 5e-7
L2_REG_WEIGHT = 1e-4






## Data Augmentation Pipeline (Albumentations) 
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
   
    # Kept basic Hue/Saturation shift
    A.HueSaturationValue(p=0.4),
    A.Normalize(),
    ToTensorV2(),
])


val_transforms = A.Compose([
    A.Normalize(),
    ToTensorV2()
])
##Custom Data Generator (tf.keras.utils.Sequence) 
def crop_face(image, detector):
    """Detects and crops the main face from an image."""
    results = detector.detect_faces(image)
    if not results:
        return None


    results.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)
    x, y, w, h = results[0]['box']
   
    margin = int(0.2 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)
   
    cropped_face = image[y1:y2, x1:x2]
   
    if cropped_face.size == 0:
        return None
       
    return cropped_face


class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, img_files, labels, batch_size, transforms, shuffle=True, detector=None):
        self.img_files = img_files
        self.labels = labels
        self.batch_size = batch_size
        self.transforms = transforms
        self.shuffle = shuffle
        self.detector = detector
        self.on_epoch_end()


    def __len__(self):
        return int(np.ceil(len(self.img_files) / self.batch_size))


    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.img_files, self.labels))
            random.shuffle(combined)
            self.img_files, self.labels = zip(*combined)
            self.img_files = list(self.img_files)
            self.labels = list(self.labels)


    def __getitem__(self, idx):
        batch_x = self.img_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
       
        imgs = []
        final_labels = []
       
        for i, path in enumerate(batch_x):
            try:
                image = cv2.imread(path)
               
                if image is None:
                    continue
                   
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               
                if self.detector:
                    image = crop_face(image, self.detector)
                    if image is None:
                        continue
                   
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
               
                augmented = self.transforms(image=image)
                img_np = np.transpose(augmented["image"].numpy(), (1, 2, 0))
               
                imgs.append(img_np)
                final_labels.append(batch_y[i])
               
            except Exception:
                continue


        if not imgs:
             return np.zeros((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)


        imgs_tensor = tf.convert_to_tensor(np.stack(imgs).astype('float32'))
        labels_tensor = tf.convert_to_tensor(np.array(final_labels), dtype=tf.float32)
       
        return imgs_tensor, labels_tensor


##  Dataset Collection Utility
def collect_dataset(root_dir):
    """Recursively collects image paths and assigns binary labels."""
    img_paths = []
    labels = []
    class_map = {'Fake': 0, 'Real': 1}
    for cls, label in class_map.items():
        folder = os.path.join(root_dir, cls)
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            files = glob.glob(os.path.join(folder, ext))
            img_paths.extend(files)
            labels.extend([label] * len(files))
    return img_paths, labels


## Model Building Function
def build_model(img_size, l2_reg_weight):
    """Builds the EfficientNetV2B3 deepfake detection model."""
    base_model = EfficientNetV2B3(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
 
    dense_output = Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.L2(l2_reg_weight),
        name='dense_features'
    )(x)
   
    x = Dropout(0.4)(dense_output)
   
    out = Dense(1, dtype="float32", name='prediction_output')(x)
   
    model = Model(inputs=base_model.input, outputs=out)


    return model, base_model


## Main Execution
if __name__ == '__main__':
    # --- Paths and Setup ---
    train_root_dir = r"C:\UECS2523_ML_Assignment_Group5\Dataset\Train"
    test_dir = r"C:\UECS2523_ML_Assignment_Group5\Dataset\Test"
    checkpoint_dir = r'C:\UECS2523_ML_Assignment_Group5\Dataset\checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)


    # --- 1. Data Collection & Stratified Splitting ---
    all_train_paths, all_train_labels = collect_dataset(train_root_dir)
    test_img_paths, test_labels = collect_dataset(test_dir)


    X_all_train = np.array(all_train_paths)
    y_all_train = np.array(all_train_labels)


    train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
        X_all_train, y_all_train,
        test_size=0.20,
        random_state=42,
        stratify=y_all_train
    )
   
    # --- 2. Class Weight Calculation ---
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights_array))
   
    print(f"Data split: {len(train_img_paths)} train, {len(val_img_paths)} val, {len(test_img_paths)} test images.")
    print(f"Computed Class weights: {class_weight_dict}")
   
    # --- Data Generators (Passing the detector) ---
    train_gen = CustomDataset(train_img_paths.tolist(), train_labels.tolist(), BATCH_SIZE, transforms=train_transforms, detector=face_detector)
    val_gen = CustomDataset(val_img_paths.tolist(), val_labels.tolist(), BATCH_SIZE, transforms=val_transforms, shuffle=False, detector=face_detector)
    test_gen = CustomDataset(test_img_paths, test_labels, BATCH_SIZE, transforms=val_transforms, shuffle=False, detector=face_detector)


    # --- Build and Compile Model (Initial Compile for Phase 1) ---
    model, base_model = build_model(IMG_SIZE, L2_REG_WEIGHT)
   
    optimizer_phase1 = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
   
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
   
    model.compile(optimizer=optimizer_phase1, loss=loss, metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])
    model.summary()
   
    # --- RESUME LOGIC ---
    CHECKPOINT_FILE = os.path.join(checkpoint_dir, 'model_weights_best.ckpt')
    initial_epoch = 0
    EPOCHS_PHASE_1 = EPOCHS
   
    if tf.io.gfile.exists(CHECKPOINT_FILE + '.index'):
        print(f"\nFound checkpoint: {CHECKPOINT_FILE}. Restoring weights for resume...")
        model.load_weights(CHECKPOINT_FILE)
        initial_epoch = 16
        EPOCHS_PHASE_1 = initial_epoch + 1
        print(f"Resuming Phase 1 training from Epoch {initial_epoch + 1}. Max limit set to {EPOCHS_PHASE_1} to force transition to Phase 2.")
    else:
        print("\nNo checkpoint found. Starting training from scratch.")


   
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
   
    checkpoint_cb = ModelCheckpoint(
        filepath=CHECKPOINT_FILE,
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=8, min_lr=1e-8, verbose=1)


    # --- Main Training Loop (Phase 1: Initial Training) ---
    print("\n--- Starting Phase 1: Initial Feature Training (Head Training) ---")
   
    # Freeze the base model entirely for Phase 1
    base_model.trainable = False
   
    history = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE_1,
        initial_epoch=initial_epoch,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=[early_stop, checkpoint_cb, reduce_lr],
        workers=NUM_WORKERS,
        use_multiprocessing=False 
    )
   
    # --- Fine-Tuning Phase (Phase 2: Pushing Accuracy to 93%+) ---
   
    print("\n--- Starting Phase 2: Fine-Tuning the Base Model ---")
   
    # 1. Load the Best Weights again
    model.load_weights(CHECKPOINT_FILE)
   
    # 2. Unfreeze Layers: Freeze the first 20% of layers, unfreeze the rest
    base_model.trainable = True
   
    fine_tune_at = int(len(base_model.layers) * 0.20)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False


    # 3. Recompile with the CRITICAL, ultra-low learning rate (5e-7)
    optimizer_phase2 = tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR)


    model.compile(
        optimizer=optimizer_phase2,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    model.summary(expand_nested=True)
   
    # 4. Run the second training loop
    history_fine_tune = model.fit(
        train_gen,
        epochs=EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history.epoch[-1] + 1,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=[early_stop, checkpoint_cb, reduce_lr],
        workers=NUM_WORKERS,
        use_multiprocessing=False 
    )
   
    # FINAL SAVE FOR DEPLOYMENT (.h5) ---
    model.load_weights(CHECKPOINT_FILE)


    final_model_h5_path = os.path.join(checkpoint_dir, 'model.h5')
    model.save(final_model_h5_path, save_format='h5')
    print(f'\nFull deployment model saved after fine-tuning to: {final_model_h5_path}')


    # Save final weights (extra backup)
    final_weights_path = os.path.join(checkpoint_dir, 'efficientnetv2b3_deepfake_model_final_weights.ckpt')
    model.save_weights(final_weights_path)
    print(f'Final model weights saved to {final_weights_path}')


    # Combine histories for plotting
    history_combined = {}
    for key in history.history.keys():
        history_combined[key] = history.history[key] + history_fine_tune.history[key]


    with open('train_history.pkl', 'wb') as f:
        pickle.dump(history_combined, f)
       
    # --- Evaluation ---
    print("\nStarting Test Evaluation...")
    test_gen.on_epoch_end()
   
    test_loss, test_acc, test_recall = model.evaluate(
        test_gen,
        workers=NUM_WORKERS,
        use_multiprocessing=False
    )
    print(f"\nFINAL TEST LOSS: {test_loss:.4f} \t FINAL TEST ACCURACY: {test_acc:.4f} \t FINAL TEST RECALL: {test_recall:.4f}")


    # Confusion matrix & report
    Y_pred_logits = model.predict(
        test_gen,
        workers=NUM_WORKERS,
        use_multiprocessing=False 
    )
    Y_pred_probabilities = tf.nn.sigmoid(Y_pred_logits).numpy()
    y_pred = (Y_pred_probabilities.flatten() > 0.5).astype(int)
   
    true_labels = []
    for i in range(len(test_gen)):
        labels_batch = test_gen[i][1].numpy().flatten()
        true_labels.extend(labels_batch)
   
    y_true = np.array(true_labels[:len(y_pred)])
   
    print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))
    print('\nClassification Report:\n', classification_report(y_true, y_pred, target_names=['Fake','Real']))


    # Plot training
    plt.figure(figsize=(12,4))
   
    # Plotting Accuracy
    plt.subplot(1,2,1)
    plt.plot(history_combined['accuracy'], label='Train Accuracy')
    plt.plot(history_combined['val_accuracy'], label='Val Accuracy')
    plt.axvline(x=len(history.history['accuracy'])-1, color='r', linestyle='--', label='Fine-Tune Start')
    plt.legend()
    plt.title('Accuracy Curve')
   
    # Plotting Loss
    plt.subplot(1,2,2)
    plt.plot(history_combined['loss'], label='Train Loss')
    plt.plot(history_combined['val_loss'], label='Val Loss')
    plt.axvline(x=len(history.history['loss'])-1, color='r', linestyle='--', label='Fine-Tune Start')
    plt.legend()
    plt.title('Loss Curve')
    plt.show()