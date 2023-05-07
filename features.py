import cv2
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim
import lpips as lpips_base
import torchvision.transforms as transforms
import erqa
from tqdm import tqdm

loss_fn_alex = None
erqa_metric = None

class FeatureExtractor:

    def __init__(self, features):
        self.features = features

        self.name2feature = {
            "ssim" : ssim_calc,
            "lpips" : lpips_calc,
            "gabor" : gabor_calc,
            "sobel" : sobel_calc,
            "lbp" : lbp_calc,
            "haff" : haff_calc,
            "sobel_sd" : sobel_sd_calc,
            "optical" : optical_calc,
            "fft" : fft_calc,
            "laplac" : laplac_calc,
            "colorfulness_diff" : color_calc,
            "colorfulness" : nr_color_calc,
            "hist" : hist_calc,
            "SI" : SI,
            "TI" : TI(),
            "erqa" : erqa_calc
        }
    

    def __call__(self, dist_imgs, ref_imgs):
        result = []
        for dist_img, ref_img in tqdm(list(zip(dist_imgs, ref_imgs))):
            result.append(self.run_on_frame(dist_img, ref_img))
        
        return self.aggregate(result)

    def run_on_frame(self, dist_img, ref_img):
        values = {}
        for feature in self.features:
            values[feature] = self.name2feature[feature](dist_img, ref_img)
        return values
    

    def reinit(self):
        self.name2feature["TI"] = TI()


    def transform(self, feat_dict):
        new_dict = {}
        for key in feat_dict:
            value = feat_dict[key]
            if type(value) is list:
                for i, elem in enumerate(value):
                    new_dict[key + "_" + str(i)] = elem
            else:
                new_dict[key] = value
        return new_dict
    

    def aggregate(self, feat_list):
        result = {}
        aggr = {}

        for feat_frame in feat_list:
            feat_frame = self.transform(feat_frame)
            for key in feat_frame:
                if key not in aggr:
                    aggr[key] = []
                aggr[key].append(feat_frame[key])
        
        for key in aggr:
            result[key] = {
                "mean" : np.mean(aggr[key]),
                "min" : min(aggr[key]),
                "max" : max(aggr[key])
            }

        return result




def lpips_calc(img1, img2):
    global loss_fn_alex
    if loss_fn_alex is None:
        loss_fn_alex = lpips_base.LPIPS(net='alex',verbose=False)
    transform = transforms.ToTensor()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = transform(img1)
    img2 = transform(img2)
    res = loss_fn_alex(img1, img2).detach().numpy()[0][0][0][0]
    return np.round(res,decimals=4)


def ssim_calc(im1, im2):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YUV)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YUV)
    Y1, U1, V1 = [im1[...,i] for i in range(3)]
    Y2, U2, V2 = [im2[...,i] for i in range(3)]
    Y = ssim(Y1, Y2)
    U = ssim(U1, U2)
    V = ssim(V1, V2)
    return [Y, U, V]


def gabor(image):
    rcs = []
    for frequency in (0.10, 0.15, 0.2):
        sigma = 3.5
        for theta in (0, np.pi / 3):
            real, _ = skimage.filters.gabor(
                image, frequency=frequency, theta=np.pi / 3, sigma_x=sigma, sigma_y=sigma, mode="wrap"
            )
            rcs.append(np.array(cv2.meanStdDev(real)))
    return rcs


def gabor_calc(im1, im2):
    """ktau=0.71
    * x4 Downsample does not change quality (-0.006)
    * frequencies (0.05, 0.10, 0.15) to 0.10 decreases quality (-0.02)
    """

    gabor_1 = gabor(cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), (128, 128)))
    gabor_2 = gabor(cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), (128, 128)))

    res = []
    for elem1, elem2 in zip(gabor_1, gabor_2):
        res.append(-np.linalg.norm(elem1 - elem2))
    return res


def sobel(image):
    grad_x = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=13)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=13)

    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)

    return grad


def sobel_calc(im1, im2):
    """Calculates norm of image edge difference
    * Large kernel size increases quality (+0.05)
    * Grayscale conversion descreases quality (-0.02)
    * Histogram of edges descreases quality (-0.04)
    * x4 Downsample decreases quality (-0.05)
    * SSIM instead of norm decreases quality (-0.12)
    """

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1 = cv2.equalizeHist(im1)
    im2 = cv2.equalizeHist(im2)

    edge_1 = sobel(im1)
    edge_2 = sobel(im2)

    return np.linalg.norm(edge_1 - edge_2)


def lbp(image):
    edges = np.rint(sobel(image)).astype(np.uint8)
    gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    patterns = skimage.feature.local_binary_pattern(gray, P=4, R=8, method='uniform')

    return patterns


def lbp_calc(im1, im2):
    """Calculates norm of image lbp descriptors difference
    * P and R tuning increases quality (+0.11)
    * Edge detectcion increases quality (+0.09)
    * Histogram calculation decreases quality (-0.06)
    """

    lbp_1 = lbp(im1)
    lbp_2 = lbp(im2)

    return -np.linalg.norm(lbp_1 - lbp_2)


def haff(img):
    edges = cv2.Canny(img, 150, 255)
    lines = cv2.HoughLinesP(edges, 200, np.pi / 3, 150, None, 0, 0)
    image = np.zeros_like(img)
    if lines is not None:
        for linee in lines:
            line = linee[0]
            cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), thickness=5)
    return image


def haff_calc(im1, im2):
    """Calculates norm of image lines difference
    * Canny threshold tuning increases quality
    * Line thickness increases quality
    """

    haff_1 = haff(im1)
    haff_2 = haff(im2)

    return np.linalg.norm(haff_1 - haff_2)


def sobel_sd(img):
    """
    Second derivative of image gradients
    """

    grad_x = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=2, dy=0, ksize=13)
    grad_y = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=2, ksize=13)

    grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
    cv2.normalize(grad, grad, 0, 255, cv2.NORM_MINMAX)

    return grad


def sobel_sd_calc(im1, im2):
    """Calculates norm of image edge difference
    * Large kernel size increases quality (+0.05)
    * Grayscale conversion descreases quality (-0.02)
    * Histogram of edges descreases quality (-0.04)
    * x4 Downsample decreases quality (-0.05)
    * SSIM instead of norm decreases quality (-0.12)
    """

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1 = cv2.equalizeHist(im1)
    im2 = cv2.equalizeHist(im2)

    edge_1 = sobel_sd(im1)
    edge_2 = sobel_sd(im2)

    return np.linalg.norm(edge_1 - edge_2)



def optical_calc(im1, im2):
    # edge_1 = np.rint(sobel(im1)).astype(np.uint8)
    # edge_2 = np.rint(sobel(im2)).astype(np.uint8)

    edge_1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YUV)[:,:,0]
    edge_2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YUV)[:,:,0]

    # flow = cv2.calcOpticalFlowFarneback(edge_1, edge_2, None, pyr_scale=0.8, levels=3, winsize=15, iterations=7, poly_n=5, poly_sigma=0, flags=0)
    flow2 = cv2.calcOpticalFlowFarneback(edge_2, edge_1, None, pyr_scale=0.8, levels=3, winsize=15, iterations=10, poly_n=5, poly_sigma=1, flags=0)

    mid = flow2[:,:,1]
    # mid = np.sqrt(np.square(flow[:,:,0]) + np.square(flow[:,:,1]))

    return np.var(mid)


def fft(image, size=35):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = np.log(np.abs(recon))
    return magnitude


def fft_calc(im1, im2):

    im1 = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), (128, 128))
    im2 = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), (128, 128))


    freqs = [30]

    sum_ = 0
    for freq in freqs:
        fft_1 = fft(im1, freq)
        fft_2 = fft(im2, freq)
        sum_ += np.linalg.norm(fft_1 - fft_2)

    return sum_


def laplac(im1):
    return cv2.Laplacian(im1, cv2.CV_64F, ksize=3)


def laplac_calc(im1, im2):

    im1 = cv2.resize(im1, (128, 128))
    im2 = cv2.resize(im2, (128, 128))

    lap_1 = laplac(im1)
    lap_2 = laplac(im2)

    return np.linalg.norm(lap_1 - lap_2)


def color(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	# split the image into its respective RGB components
    (B, G, R) = cv2.split(im.astype("float"))
	# compute rg = R - G
    rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def color_calc(im1, im2):

    c_1 = color(im1)
    c_2 = color(im2)

    return np.abs(c_1 - c_2)


def nr_color_calc(im1, im2=None):
    return color(im1)


def hist_calc(im1, im2):
    img1_hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

    hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0.1, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0.1, beta=1, norm_type=cv2.NORM_MINMAX);

    # find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
    # metric_val =  cv2.EMD(hist_img1, hist_img2, cv2.DIST_L2)[0]
    return metric_val


def SI(frame, _):
    grad_x = cv2.Sobel(frame, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=13)
    grad_y = cv2.Sobel(frame, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=13)
    value = np.hypot(grad_x, grad_y).std()
    return value


class TI:
    def __init__(self):
        self._previous_frame = None

    def __call__(self, frame, _):
        value = 0
        if self._previous_frame is not None:
            value = (frame - self._previous_frame).std()
        self._previous_frame = frame
        return value


def erqa_calc(im1, im2):
    global erqa_metric
    if erqa_metric is None:
        erqa_metric = erqa.ERQA()
    try:
        return erqa_metric(im1, im2)
    except:
        return np.nan