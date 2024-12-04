# feature_extraction.py

import numpy as np
import cv2
from scipy.ndimage import label, sobel

class FeatureExtractor:
    """
    Class for extracting features from images for digit classification.
    """

    @staticmethod
    def density(img):
        """
        Calculates the average pixel density of the image.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - float: Average pixel density.
        """
        return np.sum(img) / img.size

    @staticmethod
    def measure_symmetry(img):
        """
        Calculates the vertical symmetry of the image.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - float: Symmetry measure.
        """
        img_flipped = np.flipud(img)
        xor_img = np.bitwise_xor(img > 128, img_flipped > 128)
        symmetry = np.sum(xor_img) / img.size
        return symmetry

    @staticmethod
    def binarize_image(img, threshold=128):
        """
        Binarizes the image using a threshold.

        Params:
        - img (numpy array): Grayscale image.
        - threshold (int): Threshold value.

        Returns:
        - numpy array: Binary image.
        """
        return (img >= threshold).astype(np.uint8)

    @staticmethod
    def vertical_intersections(img):
        """
        Calculates the maximum and average vertical intersections.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - tuple: (max_intersections, avg_intersections)
        """
        bin_img = FeatureExtractor.binarize_image(img)
        intersections = []
        for col in range(bin_img.shape[1]):
            col_intersections = np.sum(bin_img[1:, col] != bin_img[:-1, col])
            intersections.append(col_intersections)
        max_intersections = max(intersections)
        avg_intersections = sum(intersections) / bin_img.shape[1]
        return max_intersections, avg_intersections

    @staticmethod
    def horizontal_intersections(img):
        """
        Calculates the maximum and average horizontal intersections.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - tuple: (max_intersections, avg_intersections)
        """
        bin_img = FeatureExtractor.binarize_image(img)
        intersections = []
        for row in range(bin_img.shape[0]):
            row_intersections = np.sum(bin_img[row, 1:] != bin_img[row, :-1])
            intersections.append(row_intersections)
        max_intersections = max(intersections)
        avg_intersections = sum(intersections) / bin_img.shape[0]
        return max_intersections, avg_intersections

    @staticmethod
    def central_pixel_density(img):
        """
        Calculates the density of the central 10x10 region.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - float: Central pixel density.
        """
        center_region = img[9:19, 9:19]
        density = np.sum(center_region) / (10 * 10)
        return density

    @staticmethod
    def number_of_loops(img):
        """
        Calculates the number of loops in the image.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - int: Number of loops.
        """
        bin_img = FeatureExtractor.binarize_image(img)
        labeled_array, num_features = label(bin_img)
        return num_features

    @staticmethod
    def aspect_ratio(img):
        """
        Calculates the aspect ratio of the digit in the image.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - float: Aspect ratio.
        """
        rows = np.any(img > 128, axis=1)
        cols = np.any(img > 128, axis=0)
        if not rows.any() or not cols.any():
            return 1.0  # Avoid division by zero
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        aspect_ratio = (cmax - cmin + 1) / (rmax - rmin + 1)
        return aspect_ratio

    @staticmethod
    def number_of_corners(img):
        """
        Calculates the number of corners in the image using Harris corner detection.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - int: Number of corners.
        """
        img_float = np.float32(img)
        dst = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        corners = dst > 0.01 * dst.max()
        num_corners = np.sum(corners)
        return num_corners

    @staticmethod
    def density_per_quadrant(img):
        """
        Calculates the density of pixels in each quadrant.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - tuple: Densities in top-left, top-right, bottom-left, bottom-right quadrants.
        """
        h, w = img.shape
        half_h, half_w = h // 2, w // 2
        # Divide the image into four quadrants
        top_left = img[:half_h, :half_w]
        top_right = img[:half_h, half_w:]
        bottom_left = img[half_h:, :half_w]
        bottom_right = img[half_h:, half_w:]
        # Calculate the density of pixels in each quadrant
        density_tl = np.sum(top_left) / top_left.size
        density_tr = np.sum(top_right) / top_right.size
        density_bl = np.sum(bottom_left) / bottom_left.size
        density_br = np.sum(bottom_right) / bottom_right.size
        return density_tl, density_tr, density_bl, density_br

    @staticmethod
    def edge_density(img):
        """
        Calculates the edge density of the image using Sobel filters.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - float: Edge density.
        """
        edges = sobel(img)
        edge_density = np.sum(edges > 0) / img.size
        return edge_density

    @staticmethod
    def extract_features(img):
        """
        Extracts all features from the image.

        Params:
        - img (numpy array): Grayscale image.

        Returns:
        - numpy array: Feature vector.
        """
        features = []
        features.append(FeatureExtractor.density(img))
        features.append(FeatureExtractor.measure_symmetry(img))
        h_max, h_avg = FeatureExtractor.horizontal_intersections(img)
        features.extend([h_max, h_avg])
        v_max, v_avg = FeatureExtractor.vertical_intersections(img)
        features.extend([v_max, v_avg])
        features.append(FeatureExtractor.number_of_loops(img))
        features.append(FeatureExtractor.number_of_corners(img))
        d_tl, d_tr, d_bl, d_br = FeatureExtractor.density_per_quadrant(img)
        features.extend([d_tl, d_tr, d_bl, d_br])
        features.append(FeatureExtractor.edge_density(img))
        features.append(FeatureExtractor.central_pixel_density(img))
        features.append(FeatureExtractor.aspect_ratio(img))
        return np.array(features)
