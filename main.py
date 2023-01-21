import imageio.v2 as imageio
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16, 10)
plt.rc('font', size=4)

classes = []  # 0, 1, 2, ...
images = []  # n*28*28
classifies = []  # [ [0,0,1,0, ....], [0,0,1,0, ....], ...]
count_classifies = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
images_class0 = []  # [image, path_name]

###################################################
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

# images = np.concatenate((images, images_class0), axis=0)
# images = np.concatenate((images, images_class0), axis=0)
# images = np.concatenate((images, images_class0), axis=0)

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
print('images_0', np.shape(images_class0))

# print('classifies', classifies[0:4])
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

# Normalization
X_train = X_train / 255.
X_train = X_train.reshape([-1, 28*28])
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int32)

X_test = X_test / 255.
X_test = X_test.reshape([-1, 28*28])
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int32)

# Visualization
labels_map = {0: 'T-Shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
              5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}
columns = 8
rows = 8
fig = plt.figure(figsize=(8, 8))

for i in range(1, columns * rows+1):
    data_idx = np.random.randint(len(X_train))
    img = X_train[data_idx].reshape([28, 28])
    label = labels_map[y_train[data_idx]]

    fig.add_subplot(rows, columns, i)
    plt.title(label)
    # plt.imshow(img, cmap='gray')
    plt.imshow(img)
    plt.axis('off')
# plt.tight_layout()

plt.show()
