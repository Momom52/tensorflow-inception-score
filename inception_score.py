import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets.inception import inception_v3
import tensorflow.contrib.eager as tfe

session = tf.InteractiveSession()

batch_size = 32

def get_inception_score(images, batch_size, splits=10):
    """
    the function is to calculate the inception score of the generated images
    image is a numpy array with values should be in the range[0, 255]

    images 299x299x3
    """
    assert(type(images) == np.ndarray)

    inception_model = inception_v3
    inception_model.eval()

    def get_softmax(x):
        x = inception_model(x)
        return tf.nn.softmax(x)

    n = len(images) // batch_size
    preds = np.zeros([len(images), 1000], dtype=np.float32)

    tfe.enable_egaer_execution()
    dataloader = tf.data.Dataset.from_tensor_slices(images)
    dataloader = data.batch(batch_size)
    for i, batch in enumerate(tfe.Iterator(dataloader), 0):
        batch_x = tf.Variable(batch) # images
        # softmax
        preds[i * batch_size:(i + 1) * batch_size] = get_softmax(batch_x)

    scores = []
    # IS score
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)



if __name__ == '__main__':

    score = get_inception_score(images)
    print(score)





####the bull shit