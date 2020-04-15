import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import os
import shutil
import statistics
# import pandas as pd

# for printing unlimited np array elements (to max size)
np.set_printoptions(threshold=np.inf)

image_name = '../images/index.jpeg'
# contingent on the image not in the current dir
# make sure that the image is either in any of the parent or child directories
last_slash_index = image_name[::-1].find('/')
last_dot_index = image_name[::-1].find('.')
last_slash_index = -last_slash_index - 1
last_dot_index = -last_dot_index - 1
make_dir = image_name[last_slash_index + 1: last_dot_index]

image = cv2.imread(image_name, 0)
#TODO resizing not needed, work with default image size
# image = cv2.resize(image, (750, 750))
rows = image.shape[0]
cols = image.shape[1]

# plot original image
plt.subplot(121)
plt.imshow(image, cmap = 'gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

# making mid white pure white and mid black pure black
#TODO find optimal values of these thresholds
'''
image[image > 120] = 255
image[image < 80] = 0
'''
# product with 2 and 4 concluded to be good enough thresholds by observations
# for different input images
mean = int(np.mean(image))
std_dev = int(np.std(image))
image[image > mean - 2 * std_dev] = 255
image[image < mean - 4 * std_dev] = 0

# thresholding methods
'''
image = cv2.medianBlur(image, 5)
# ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
'''

# plot intensified white black image
plt.subplot(122)
plt.imshow(image, cmap = 'gray')
plt.title('Zeroed Image')
plt.xticks([])
plt.yticks([])
plt.show()

# line range array
indexes = []

# extract lines
# use row_sum to not consider spurious rows with only a few non-white pixels
white_thresh = cols * 250
row_sum = image.sum(1)
start = -1
stop = -1
for i in range(0, rows):
    if row_sum[i] >= white_thresh:
    # if np.all(image[i]):
        if start != -1:
            stop = i
            indexes.append((start, stop))
            start = -1
            stop = -1
    else:
        if start == -1:
            start = i

# lines image matrices array
lines = []

# stores line lengths
line_lengths = []

# compute average line length
for limits in indexes:
    line_lengths.append(limits[1] - limits[0])
avg_line_length = statistics.mean(line_lengths)

# store line image matrices
# remove spurious lines
# increase range of rest of the lines
for line_range in indexes:
    if line_range[1] + 1 - line_range[0] > 10:
        image_temp = image[line_range[0] - 5:line_range[1] + 1 + 5]
        lines.append(image_temp)
    '''
    if line_range[1] + 1 - line_range[0] > avg_line_length:
        # 10% padding for the line above and below
        padding = int(0.1 * (line_range[1] - line_range[0]))
        lines.append(image[line_range[0] - padding:line_range[1] + padding + 1])
    # adding white border around the extracted lines
    '''
    '''
    image_temp = image[line_range[0] - 5:line_range[1] + 1 + 5]
    top = int(0.1 * (line_range[1] - line_range[0]))
    image_temp = cv2.copyMakeBorder(image_temp, top, top, top, top,
            cv2.BORDER_CONSTANT, value=[255, 255, 255])
    lines.append(image_temp)
    '''

# plot lines
plot_num = 1
num_of_lines = len(lines)
for line in lines:
    plt.subplot(num_of_lines, 1, plot_num)
    plt.imshow(line, cmap = 'gray')
    plt.title('line ' + str(plot_num))
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(hspace=1)
    plot_num += 1
plt.show()

# words' ranges in individual lines
words_in_lines = []

# extract words
for line in lines:
    start = -1
    stop = -1
    line_t = line.T
    words = []
    for i in range(0, line_t.shape[0]):
        if np.all(line_t[i]):
            if start != -1:
                stop = i
                words.append((start, stop))
                start = -1
                stop = -1
        else:
            if start == -1:
                start = i
    words_new = copy.deepcopy(words)
    words_in_lines.append(words_new)
    words.clear()

# spaces' counts in individual lines
spaces_for_lines = []

# extract space counts
for words in words_in_lines:
    spaces = []
    for i in range(1, len(words)):
        spaces.append(words[i][0] - words[i - 1][1])
        spaces_new = copy.deepcopy(spaces)
    spaces_for_lines.append(spaces_new)

# calculating mean and standard deviation of all spaces and storing in
# respective lists
avg_space_lengths = []
# std_dev_space_lengths = []
for spaces in spaces_for_lines:
    mean = statistics.mean(spaces)
    # std_dev = statistics.stdev(spaces)
    avg_space_lengths.append(mean)
    # std_dev_space_lengths.append(std_dev)

# print(words_in_lines, spaces_for_lines, avg_space_lengths, std_dev_space_lengths)
# print(words_in_lines, spaces_for_lines, avg_space_lengths)

# detect space length and merge extracted words with gaps less than this length
# between them
for i in range(0, len(spaces_for_lines)):
    for j in range(len(spaces_for_lines[i]) - 1, -1, -1):
        #TODO find optimal value for this space length check
        if spaces_for_lines[i][j] <= avg_space_lengths[i]:
            words_in_lines[i][j] = (words_in_lines[i][j][0], words_in_lines[i][j
                + 1][1])
            del words_in_lines[i][j + 1]

# print(words_in_lines)

# check if directory for input image already exists to save words
if make_dir not in os.listdir('../generated_words/'):
    os.mkdir('../generated_words/' + make_dir)
else:
    shutil.rmtree('../generated_words/' + make_dir)
    os.mkdir('../generated_words/' + make_dir)
# check if directory for input image already exists to save lines
if make_dir not in os.listdir('../generated_lines/'):
    os.mkdir('../generated_lines/' + make_dir)
else:
    shutil.rmtree('../generated_lines/' + make_dir)
    os.mkdir('../generated_lines/' + make_dir)

# saving resized 128 * 32 word images of all lines since these dimensions are
# needed by the NN
# saving line images
step = 1
for words, line in zip(words_in_lines, lines):
    plot_num = 1
    num_of_words = len(words)
    for word in words:
        plt.subplot(1, num_of_words, plot_num)
        plt.imshow(line[:, word[0]:word[1] + 1], cmap='gray')
        plt.title('word ' + str(plot_num))
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=1)
        cv2.imwrite('../generated_words/' + make_dir + '/line_' + str(step) +
                '_word_' + str(plot_num) + '.jpg', cv2.resize(line[:, word[0]:word[1] +
                    1], (128, 32)))
        plot_num += 1
        cv2.imwrite('../generated_lines/' + make_dir + '/line_' + str(step) +
                '.jpg', line)
    step += 1
    plt.show()

# testing file inputs to be provided in specfic order to the NN
'''
for filename in sorted(os.listdir('generated_words/para_6_with_gaps/')):
    if filename.endswith('.jpg'):
        print(filename)
'''

# testing
'''
np.savetxt("image_pixels.txt", image, fmt="%d")
pd.DataFrame(image).to_csv('image.csv')

with open("image_pixels.txt", "w") as f:
    for line in image:
        # np.savetxt(f, line, fmt='%d')
        # line = str(line).replace('[', '')
        # line.replace(']', '')
        # f.write(line)
'''
