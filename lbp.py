import cv2
import numpy as np
from matplotlib import pyplot as plt
import hog
import argparse

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] > center:
            new_value = 1
    except:
        pass
    return new_value

def lbp(img, x, y):
    '''
         64 | 128 |   1
        ----------------
         32 |   0 |   2
        ----------------
         16 |   8 |   4
    '''
    center = img[x][y]
    neighbor_val = []
    neighbor_val.append(get_pixel(img, center, x-1, y+1))   # top_right
    neighbor_val.append(get_pixel(img, center, x, y+1))   # right
    neighbor_val.append(get_pixel(img, center, x+1, y+1))   # bottom_right
    neighbor_val.append(get_pixel(img, center, x+1, y))   # bottom
    neighbor_val.append(get_pixel(img, center, x+1, y-1))   # bottom_left
    neighbor_val.append(get_pixel(img, center, x, y-1))   # left
    neighbor_val.append(get_pixel(img, center, x-1, y-1))   # top_left
    neighbor_val.append(get_pixel(img, center, x-1, y))   # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(neighbor_val)) :
        val += neighbor_val[i] * power_val[i]
    return val

def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i + 1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap=plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color="black")
            current_plot.set_xlim([0, 260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list, rotation=90)

    plt.show()

def main():
    # read the image and make it gray
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    height, width, channel = image.shape
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # execute LBP
    img_lbp = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp(img_gray, i, j)
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    output_list = []
    output_list.append({
        "img": img_gray,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "Gray Image",
        "type": "gray"
    })
    output_list.append({
        "img": img_lbp,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "LBP Image",
        "type": "gray"
    })
    output_list.append({
        "img": hist_lbp,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)",
        "type": "histogram"
    })

    # execute HOG
    horizontal_mask = np.array([-1, 0, -1])
    vertical_mask = np.array([[-1],
                              [0],
                              [1]])

    horizontal_grad = hog.calculate_gradient(img_gray, horizontal_mask)
    vertical_grad = hog.calculate_gradient(img_gray, vertical_mask)

    grad_magnitude = hog.gradient_magnitude(horizontal_grad, vertical_grad)
    grad_direction = hog.gradient_direction(horizontal_grad, vertical_grad)

    grad_direction = grad_direction % 180
    hist_bins = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

    # histogram of the first cell in the first block
    cell_direction = grad_direction[:8, :8]
    cell_magnitude = grad_magnitude[:8, :8]
    HOG_cell_hist = hog.hog(cell_direction, cell_magnitude, hist_bins)

    # show output of descriptors
    show_output(output_list)

    plt.bar(x=np.arange(9), height=HOG_cell_hist, align="center", width=0.8)
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()