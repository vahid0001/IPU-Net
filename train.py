import tensorflow as tf
import os
import time
import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


BUFFER_SIZE = 1000
BATCH_SIZE = 64
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3
EPOCHS = 50
BETA = 5
FRNet_PATH = 'facenet.h5'
PATH_TRAIN = "preprocessed_vggface2"
log_dir = "logs/"
checkpoint_dir = 'checkpoints'


def load(image_file):
    input_image = tf.io.read_file(image_file)
    input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.cast(input_image, tf.float32)

    return input_image


def resize(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.BICUBIC)

    return input_image


# normalizing the images to [-1, 1]
def normalize(input_image):
    return (input_image / 127.5) - 1


# pre-whitening the images
def prewhiten_face(input_image):
    mean = tf.keras.backend.mean(input_image)
    std = tf.keras.backend.std(input_image)

    return (input_image - mean) / std


def load_image_train(image_file):
    input_image = load(image_file)
    face_normalized = normalize(resize(input_image, IMG_HEIGHT, IMG_WIDTH))
    face_prewhitened = prewhiten_face(resize(input_image, 160, 160))

    return face_normalized, face_prewhitened


def load_image_test(image_file):
    input_image = load(image_file)
    face_normalized = normalize(resize(input_image, IMG_HEIGHT, IMG_WIDTH))
    face_prewhitened = prewhiten_face(resize(input_image, 160, 160))

    return face_normalized, face_prewhitened


def lr_preprocessing(input_image):
    random = tf.random.uniform((), minval=0, maxval=1000)

    if random < 100:
        input_image = resize(input_image, 7, 6)
    elif random >= 100 and random < 200:
        input_image = resize(input_image, 11, 8)
    elif random >= 200 and random < 300:
        input_image = resize(input_image, 14, 12)
    elif random >= 300 and random < 400:
        input_image = resize(input_image, 16, 12)
    elif random >= 400 and random < 500:
        input_image = resize(input_image, 16, 14)
    elif random >= 500 and random < 600:
        input_image = resize(input_image, 16, 16)
    elif random >= 600 and random < 700:
        input_image = resize(input_image, 18, 16)
    elif random >= 700 and random < 800:
        input_image = resize(input_image, 21, 15)
    elif random >= 800 and random < 900:
        input_image = resize(input_image, 32, 32)
    elif random >= 900 and random < 1000:
        input_image = resize(input_image, 112, 96)

    input_image = resize(input_image, IMG_HEIGHT, IMG_WIDTH)

    return input_image


def downsample(filters, size, strides=2, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, strides=2, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def SRNet():
    inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, CHANNELS])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def loss(hr_face, sr_face, hr_embedding, sr_embedding):
    # MSE between faces in pixel space
    pixel_loss = mse(hr_face, sr_face)

    # MSE between faces in embedding space
    embedding_loss = mse(hr_embedding, sr_embedding)

    return pixel_loss, embedding_loss


@tf.function
def train_step(hr_face, hr_embedding, sr_embedding, epoch):
    with tf.GradientTape() as gen_tape:
        sr_face = srnet(lr_preprocessing(hr_face), training=True)

        pixel_loss, embedding_loss = loss(hr_face, sr_face, hr_embedding, sr_embedding)
        total_loss = BETA * pixel_loss + embedding_loss

        srnet_gradients = gen_tape.gradient(total_loss,
                                            srnet.trainable_variables)

        srnet_optimizer.apply_gradients(zip(srnet_gradients,
                                            srnet.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('total_loss', total_loss, step=epoch)
            tf.summary.scalar('pixel_loss', pixel_loss, step=epoch)
            tf.summary.scalar('embedding_loss', embedding_loss, step=epoch)


def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()

        print("Epoch: ", epoch + 1)

        # Train
        for n, (face_normalized, face_prewhitened) in train_ds.enumerate():
            print(n.numpy() + 1)

            hr_embedding = frnet.predict(face_prewhitened)
            sr_face = srnet(lr_preprocessing(face_normalized), training=True)

            sr_face_recoverd = sr_face * 0.5 + 0.5
            sr_face_recoverd = sr_face_recoverd * 255.
            sr_face_recoverd = resize(sr_face_recoverd, 160, 160)
            sr_face_prewhitened = prewhiten_face(sr_face_recoverd)

            sr_embedding = frnet.predict(sr_face_prewhitened)
            train_step(face_normalized, hr_embedding, sr_embedding, epoch)

        # saving (checkpoint) the model each epoch
        checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


"""Input Pipeline"""
train_dataset = tf.data.Dataset.list_files(PATH_TRAIN + '/*/*.*')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

"""Building the model"""
frnet = tf.keras.models.load_model(FRNet_PATH)
srnet = SRNet()

"""Define Loss and Optimizer"""
mse = tf.keras.losses.MeanSquaredError()
srnet_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(srnet_optimizer=srnet_optimizer,
                                 srnet=srnet)

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""Train the Model"""
fit(train_dataset, EPOCHS)