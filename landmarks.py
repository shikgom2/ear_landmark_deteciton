import matplotlib.pyplot as plt

def put_landmarks(prediction_x, prediction_y, datatitle, single_img_path):

    img_result_path = "./result/" + datatitle + "_result.png"

    img_original = plt.imread(single_img_path)

    for j in range(0,55):  # drop the landmark points on the image
        plt.scatter([prediction_x[j]], [prediction_y[j]], s = 150)


    plt.imshow(img_original)
    plt.savefig(img_result_path)
    plt.close()