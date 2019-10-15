# Detecting Tennis Balls

I made this program for my trial project when I was testing out to join
the intellient systems subdivision for the robotics team at SJSU. The purpose
of this project was to ensure that we were able to use computer vision to detect
tennis balls.

## How to run the application

1. Download opencv for python by using the "pip install opencv-python" command
2. Clone/download source files, this includes the data folder as well
3. Navigate to the files and run it using your IDE
4. To test out the different test cases, replace the "batchtwo2.JPEG" on line 6
   with another file from the data folder.

## How the program works

### Core Functionality

The method I used to detect tennis balls was to use their color in my favor.
To begin, first I apply an HSV filter on the image and mask it such that the
only color that remains is the green of the tennis ball. To do this, I use two
numpy array that hold the lower and upper bound of the color of the green of the
tennis balls.

```Python
low_green = np.array([28, 47, 47])
high_green = np.array([47, 160, 255])

hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)  # convert to HSV
# makes it so we only see the green color that we changed
mask = cv.inRange(hsv, low_green, high_green)
```

I use the Gaussian Blur to sharpen the images for edges. Intuitively, this
doesn't make sense since blurring an image should make it harder to see the
indivual objects of it. But, this actually smoothens the image overall, and the
edges become more defined inbetween different objects.

```Python
blurred = cv.GaussianBlur(image, (11, 11), 0)
```

I then filter out false positives from the image by utilizing the erode and
dilate functions in CV. The erode function in OpenCV removes the small little
blobs in the image that remain after we mask the image, and the dilate function
is to recover the lost data that is relevant to us.

```Python
mask = cv.erode(mask, None, iterations=2)
# removes most of the little blobs in the picture
mask = cv.dilate(mask, None, iterations=2)
```

### Extra step

I did an extra step for when I was debugging my program to recover the colors of
my image by using the bitwise_and function of OpenCV. This isn't necessary but
it helps bridge the gap between what the computer sees and what we would see
by bringing back the color in the filtered image.

### Core Logic

The brains of this code is in the HoughCircles function. This is also a provided
function of OpenCV, and this basically uses Canny Edge detection specifically
for determing circles. After finding the cirlces, I make this into a numpy
array that I can use to draw circles. I iterate through this array and draw the
center and the radius of the circle around it.

```Python
circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, 20, param1=20, param2=10, minRadius=0, maxRadius=100)

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv.circle(image, (i[0], i[1]), i[2], (0, 255, 255), 2)
    print("x: ", i[0], " y: ", i[1], " d: ", i[2])
    cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
```

## Caveats

My program fails when a tennis ball is not in the correct color range, if it is
too big, too small, and generates multiple false positives since the color range
is so wide.

I could change a few things to the program to aleviate these issues such as
narrowing or changing my upper/lower bound for what I define as green or
changing my min/max radius for the HoughCircles function. But, ultimately, this
is not the smart way to do things.

## Future Developments

The smartest way to aleviate these issues would be to develop a neural network
and train a model that would detect the tennis balls.
