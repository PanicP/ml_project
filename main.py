import imageio.v2 as imageio
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

plt.rcParams['figure.figsize'] = (16, 10)
plt.rc('font', size=7)

classes = []  # 0, 1, 2, ...
images = []  # n*28*28
classifies = []  # [ [0,0,1,0, ....], [0,0,1,0, ....], ...]
count_classifies = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
images_class0 = []  # [image, path_name]

# Retreive all images from the folder
for im_path in glob.glob("FashionMNIST/*.png"):
    im = imageio.imread(im_path)

    # Add class
    classes.append(im_path[13])

    # Add image
    images.append(im)

    # Add count classify
    count_classifies[int(im_path[13])] += 1

    # Add classify
    classify = np.zeros(10)
    classify[int(im_path[13])] = 1
    classifies.append(classify)
    # path_names.append(im_path)
    # list of only 0
    if (int(im_path[13]) == 0):
        images_class0.append([im, im_path])

# Normalize class-0 images
while count_classifies[0] < count_classifies[1]:
    for image0 in images_class0:
        # print(count_classifies[int(image0[1][13])])
        if (count_classifies[0] >= count_classifies[1]):
            break
        # Add class
        classes.append(image0[1][13])

        # Add image
        images.append(image0[0])

        # Add count classify
        count_classifies[int(image0[1][13])] += 1

        # Add classify
        classify = np.zeros(10)
        classify[int(image0[1][13])] = 1
        classifies.append(classify)

print('images', np.shape(images))
print('classes', np.shape(classes))
print('count', count_classifies)
# print('images_0', np.shape(images_class0))

# Normalize yellow dot
for image in images:
    image[14, 14] = image[13, 14]


###################################################
# numbers = np.arange(0, 10, 1)
# print(numbers.shape, classifies.count)
# f, ax = plt.subplots(1, 1)
# ax.scatter(numbers, classifies)
# plt.show()

# Check Data
# print(images[0])
# print(classes[0])
# print('images', np.shape(images))
# print('classes', np.shape(classes))

X_train = np.array(images)
X_test = np.array(images)
y_train = np.array(classes)
y_test = np.array(classes)

# Reshaping
X_train = X_train / 255.
X_train = X_train.reshape([-1, 28*28])
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32)

X_test = X_test / 255.
X_test = X_test.reshape([-1, 28*28])
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int32)

# Visualization
# labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
#               5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}
# columns = 8
# rows = 8
# fig = plt.figure(figsize=(8, 8))

# for i in range(1, columns * rows+1):
#     data_idx = np.random.randint(len(X_train))
#     img = X_train[data_idx].reshape([28, 28])
#     label = labels_map[y_train[data_idx]]

#     fig.add_subplot(rows, columns, i)
#     plt.title(label)
#     # plt.imshow(img, cmap='gray')
#     plt.imshow(img)
#     plt.axis('off')
# plt.tight_layout()

# plt.show()


# train dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
    .shuffle(buffer_size=len(X_train))\
    .batch(batch_size=128)\
    .prefetch(buffer_size=128)\
    .repeat()

# test dataset
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))\
            .batch(batch_size=128)\
            .prefetch(buffer_size=128)\
            .repeat()

# running model

model = tf.keras.Sequential(name='nn')

model.add(tf.keras.layers.Dense(256, input_shape=(28*28, )))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.ReLU())
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.add(tf.keras.layers.Dense(10))
# model.add(tf.keras.layers.ReLU())
# model.add(tf.keras.layers.Dense(10))
# model.add(tf.keras.layers.ReLU())
# model.add(tf.keras.layers.Dense(10))
# model.add(tf.keras.layers.ReLU())
# model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Softmax())

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
#   loss=tf.keras.losses.CategoricalCrossentropy(),
#   metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(train_ds, batch_size=64,
          steps_per_epoch=len(X_train)/128, epochs=20)

loss, acc = model.evaluate(test_ds, steps=len(X_test)/128)
print('test loss is {}'.format(loss))
print('test accuracy is {}'.format(acc))

# Visualize the performance

labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
              5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

test_batch_size = 25
batch_index = np.random.choice(
    len(X_test), size=test_batch_size, replace=False)

batch_xs = X_test[batch_index]
batch_ys = y_test[batch_index]
y_pred_ = model(batch_xs, training=False)

fig = plt.figure(figsize=(10, 10))
for i, (px, py, y_pred) in enumerate(zip(batch_xs, batch_ys, y_pred_)):
    p = fig.add_subplot(5, 5, i+1)
    if np.argmax(y_pred) == py:
        p.set_title("{}".format(labels_map[py]), color='blue')
    else:
        p.set_title("{}/{}".format(labels_map[np.argmax(y_pred)],
                                   labels_map[py]), color='red')
    p.imshow(px.reshape(28, 28))
    p.axis('off')
plt.tight_layout()
plt.show()
