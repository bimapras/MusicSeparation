This project is inspired by Music Source Separation research in time domain. It draws upon several key references that have significantly influenced its development, including:
- [Demucs v2](https://arxiv.org/abs/1911.13254)  
- [Wave-U-Net](https://arxiv.org/abs/1806.03185)  
- [Conv-TasNet](https://arxiv.org/abs/1809.07454)  
- [DPRNN (Dual-Path RNN)](https://arxiv.org/abs/1910.06379)

These models serve as the main inspiration and foundational references for the design and implementation of this project.

Demo using google colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/yourusername/audio-separation/blob/main/notebooks/demo.ipynb)
# How to use
- clone repository
    ```
    git clone https://github.com/bimapras/MusicSeparation.git
    ```
- Install library and depedencies with specific version to use pretrained model
    ```
    pip install requirement.txt
    ```
    or
    ```
    pip install tensorflow=2.17 numpy musdb librosa soundfile
    ```
- Infference
    ```
    from utils import read_audio, inference
    '''
    Inference expect tflite model with input shape (time, 2)
    Make sure segment_length is your time length input model (Default model use 88064)
    '''

    reader = read_audio.AudioReader()
    audio_data, samplerate = reader.read(r'sample\Pierce The Veil - So Far So Fake (Visualizer).mp4')

    tflite_model_path = r'models/DPTCN.tflite'
    inference = inference.AudioInference(model=tflite_model_path,
                                segment_length=88064,
                                overlap=0.5,
                                batch_size=4,
                                use_wiener=True,
                                stft_frame_length=4096,
                                stft_frame_step=1024,
                                wiener_iterations=3)
    pred = inference.predict(audio_data,
                            segment_length_sec=30,
                            export=True, 
                            export_dir="DPTCN_result")
    ```

# Training your own model
### 1. Prepare Dataset MUSDB18
```
import musdb
from modules.dataset import create_wav

mus_train = musdb.DB(root='path_musdb_audio', subsets="train", split='train')

create_wav(train, mus_train, 44100, 88064)
```
You can check tutorial to use parser musdb in here [Python Parser MUSDB18](https://github.com/sigsep/sigsep-mus-db)

### 2. Create training pipeline using dataset.py
```
from modules.dataset import CachedStemMixTFDataset

BATCH_SIZE = 32

train_path = list_stem_files('prepared_audio_path', ['vocals', 'drums', 'bass', 'other'])
train_path_tensor = tf.convert_to_tensor(train_path, dtype=tf.string)

# Create training dataset class
train_builder = CachedStemMixTFDataset(
    all_paths_tensor=train_path_tensor,
    stems=('vocals', 'drums', 'bass', 'other'),
    swap_prob=0.5
)
# Note : Set swap_prob 0.0 for validation or test

train_set = train_builder.get_tf_dataset(batch_size=BATCH_SIZE, shuffle=True, augment=True)
# Note : Dont set augment & shuffle True for validation or test
```
### 3. Create model and configuration training
```
import tensorflow as tf
from modules import loss, model

loss = loss.new_demucs_loss
metrics = loss.compute_sdr
EPOCH = 80

# Create Model
model = model.SeparatorModel(
    input_length=88064,
    n_filter=64,
    n_kernel=8,
    stride=4,
    t_layer=3,
    n_chunk=4,
    bias=False,
    n_stems=4,
    n_depth=4
)
'''
you can create your own model with modify SeparatorModel in model.py
for custom layer i already put in layers_wrapper.py
'''

# Compile
model.compile(optimizer = tf.keras.optimizers.AdamW(learning_rate = 0.0003, global_clipnorm=1.0), 
              loss = loss, 
              metrics = [metrics])

# Training
history = model.fit(train_set, validation_data = val_set, epochs = EPOCH, verbose = 1,
         callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras', 
                save_best_only=True, monitor='val_compute_sdr', 
                mode='max')])
```