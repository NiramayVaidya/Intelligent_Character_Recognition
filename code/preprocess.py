import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import os
import shutil
import statistics

def save_words_and_lines(image_file):
    # contingent on the image not in the current dir
    # make sure that the image is either in any of the parent or child directories
    last_slash_index = image_file[::-1].find('/')
    last_dot_index = image_file[::-1].find('.')
    last_slash_index = -last_slash_index - 1
    last_dot_index = -last_dot_index - 1
    make_dir = image_file[last_slash_index + 1: last_dot_index]
    
    image = cv2.imread(image_file, 0)
    rows = image.shape[0]
    cols = image.shape[1]
    
    # plot original image
    '''
    plt.subplot(121)
    plt.imshow(image, cmap = 'gray')
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    '''
    
    # product with 2 and 4 concluded to be good enough thresholds by observations
    # for different input images
    mean = int(np.mean(image))
    std_dev = int(np.std(image))
    image[image > mean - 2 * std_dev] = 255
    image[image < mean - 4 * std_dev] = 0
    '''
    image[image > 150] = 255
    image[image < 80] = 0
    '''
    
    # plot intensified white black image
    '''
    plt.subplot(122)
    plt.imshow(image, cmap = 'gray')
    plt.title('Zeroed Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    '''
    
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
        '''
        if line_range[1] + 1 - line_range[0] > avg_line_length:
            # 10% padding for the line above and below
            padding = int(0.1 * (line_range[1] - line_range[0]))
            lines.append(image[line_range[0] - padding:line_range[1] + padding + 1])
        '''
        if line_range[1] + 1 - line_range[0] > 10:
            image_temp = image[line_range[0] - 5:line_range[1] + 1 + 5]
            lines.append(image_temp)
        # adding white border around the extracted lines
        '''
        image_temp = image[line_range[0] - 5:line_range[1] + 1 + 5]
        top = int(0.1 * (line_range[1] - line_range[0]))
        image_temp = cv2.copyMakeBorder(image_temp, top, top, top, top,
                cv2.BORDER_CONSTANT, value=[255, 255, 255])
        lines.append(image_temp)
        '''
        
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

    # detect space length and merge extracted words with gaps less than this length
    # between them
    for i in range(0, len(spaces_for_lines)):
        for j in range(len(spaces_for_lines[i]) - 1, -1, -1):
            #TODO find optimal value for this space length check
            if spaces_for_lines[i][j] <= avg_space_lengths[i]:
                words_in_lines[i][j] = (words_in_lines[i][j][0], words_in_lines[i][j
                    + 1][1])
                del words_in_lines[i][j + 1]


    # check if directory for input image already exists to save words
    if make_dir not in os.listdir('generated_words/'):
        os.mkdir('generated_words/' + make_dir)
    else:
        shutil.rmtree('generated_words/' + make_dir)
        os.mkdir('generated_words/' + make_dir)
    # check if directory for input image already exists to save lines
    if make_dir not in os.listdir('generated_lines/'):
        os.mkdir('generated_lines/' + make_dir)
    else:
        shutil.rmtree('generated_lines/' + make_dir)
        os.mkdir('generated_lines/' + make_dir)

    # saving resized 128 * 32 word images of all lines since these dimensions are
    # needed by the NN
    # saving line images
    step = 1
    for words, line in zip(words_in_lines, lines):
        plot_num = 1
        num_of_words = len(words)
        for word in words:
            cv2.imwrite('generated_words/' + make_dir + '/line_' + str(step) +
                    '_word_' + str(plot_num) + '.jpg', cv2.resize(line[:, word[0]:word[1] +
                        1], (128, 32)))
            plot_num += 1
            cv2.imwrite('generated_lines/' + make_dir + '/line_' + str(step) +
                    '.jpg', line)
        step += 1
    return make_dir
