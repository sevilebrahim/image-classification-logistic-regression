
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image


img = Image.open(r"D:\xxxx\dataset.bmp\1.png")
img = img.resize((64, 64))
img = img.convert("RGB")

image_array = np.array(img)

print(image_array[10][6][1])

plt.imshow(image_array)
plt.show()

x = image_array.flatten()

print(x.shape)
print(x)

image_red = image_array[:, :, 0]
image_reshaped = image_array.reshape(12288)

print(image_array.shape)
print(image_red[10][6])


a = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
])

b = a.reshape(9)
c = b.reshape(3, 3)

print(a)
print(a.reshape(9))
print(b)
print(c)


x = []

for i in range(1, 11):

    image = Image.open(
        fr"D:\xxxx\dataset.bmp\{i}.png"
    )

    image = image.resize((64, 64))
    image = image.convert("RGB")

    image_array = np.array(image)

    red_channel = image_array[:, :, 0]
    red_channel = red_channel.flatten()
    red_channel = red_channel / 255

    x.append(red_channel)

x = np.array(x)

print(x.shape)
print(x)

print(x[0].shape)

plt.imshow(
    x[8].reshape(64, 64),
    cmap="gray"
)

plt.axis("off")
plt.show()


y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])

print("X shape:", x.shape)
print("Y shape:", y.shape)

print("Number of images:", x.shape[0])
print("Number of labels:", len(y))


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression()

model.fit(
    x_train,
    y_train
)


print("Training data:", x_train.shape)
print("Test data:", x_test.shape)
print("Training labels:", y_train)
print("Test labels:", y_test)


plt.imshow(
    x_train[4].reshape(64, 64),
    cmap="gray"
)

plt.show()


out = model.predict(x_test)

print("Test predictions:", out)

plt.imshow(
    x_test[0].reshape(64, 64),
    cmap="gray"
)

plt.show()


input_image = Image.open(
    r"D:\xxxx\dataset.bmp\11.png"
)

input_array = np.array(input_image)

print("Original shape:", input_array.shape)

plt.imshow(input_array)
plt.show()


input_image = Image.open(
    r"D:\xxxx\dataset.bmp\11.png"
)

input_image = input_image.resize((64, 64))
input_image = input_image.convert("RGB")

input_array = np.array(input_image)

input_array = input_array[:, :, 0]

input_array = input_array.flatten()
input_array = input_array / 255

print("Input shape:", input_array.shape)

prediction = model.predict([input_array])

print("Prediction:", prediction)
