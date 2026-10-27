#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <optional>
#include <string>
#include <vector>


/**
 * Plain checkerboard of squares_x by squares_y squares, square (i, j) white when i + j is even
 * (i counts along x, j along y).
 *
 * Board frame: origin at the outer corner of square (0, 0), x along the squares_x direction, y
 * along squares_y, z = x cross y, which points into the printed face (away from a camera that
 * sees the board). Interior corner (c, r), c = 0..squares_x-2, r = 0..squares_y-2, sits at
 * ((c + 1) s, (r + 1) s, 0) and is stored row-major, the order OpenCV's detectors return.
 */
struct Checkerboard {
    int squares_x;
    int squares_y;
    double square_size;

    cv::Size pattern_size() const { return {squares_x - 1, squares_y - 1}; }

    /// A 180 degree turn swaps the colours only when squares_x + squares_y is odd.
    bool colours_fix_orientation() const { return (squares_x + squares_y) % 2 == 1; }

    std::vector<cv::Point3f> corners() const {
        std::vector<cv::Point3f> points;
        for (int r = 0; r < squares_y - 1; ++r)
            for (int c = 0; c < squares_x - 1; ++c)
                points.emplace_back((c + 1) * square_size, (r + 1) * square_size, 0.0f);
        return points;
    }
};


/**
 * Mean intensity of the squares that should be white minus those that should be black, sampled
 * off-centre in every square that lies in the image (centres may carry orientation dots).
 * Positive when the corner order matches the board's colouring, negative when it is turned 180
 * degrees.
 */
inline double colour_contrast(
    const cv::Mat& gray,
    const Checkerboard& board,
    const std::vector<cv::Point2f>& corners)
{
    // Homography from interior-corner grid units (corner (c, r) at (c, r)) to pixels
    std::vector<cv::Point2f> grid;
    for (int r = 0; r < board.squares_y - 1; ++r)
        for (int c = 0; c < board.squares_x - 1; ++c)
            grid.emplace_back(c, r);
    cv::Mat H = cv::findHomography(grid, corners);

    std::vector<cv::Point2f> samples;
    std::vector<int> parity;
    for (int i = 0; i < board.squares_x; ++i) {
        for (int j = 0; j < board.squares_y; ++j) {
            for (float dx : {-0.3f, 0.3f}) {
                for (float dy : {-0.3f, 0.3f}) {
                    samples.emplace_back(i - 0.5f + dx, j - 0.5f + dy);   // square (i, j) centre is (i - 0.5, j - 0.5)
                    parity.push_back((i + j) % 2);
                }
            }
        }
    }
    std::vector<cv::Point2f> pixels;
    cv::perspectiveTransform(samples, pixels, H);

    double sum[2] = {0, 0};
    int count[2] = {0, 0};
    for (size_t k = 0; k < pixels.size(); ++k) {
        const cv::Point p(cvRound(pixels[k].x), cvRound(pixels[k].y));
        if (p.x < 0 || p.y < 0 || p.x >= gray.cols || p.y >= gray.rows) continue;
        sum[parity[k]] += gray.at<uint8_t>(p);
        ++count[parity[k]];
    }
    if (count[0] == 0 || count[1] == 0) return 0.0;
    return sum[0] / count[0] - sum[1] / count[1];
}


/**
 * Detect the interior corners and put them in board order.
 *
 * findChessboardCornersSB (sub-pixel already), then the colour check turns a 180 degree
 * reversed order around. A mirrored order cannot be told from the colours; the caller rejects
 * it from the pose (board normal facing away from the camera).
 *
 * @param why Set to the reason when nothing is returned.
 */
inline std::optional<std::vector<cv::Point2f>> detect_checkerboard(
    const cv::Mat& gray,
    const Checkerboard& board,
    double min_contrast,
    std::string& why)
{
    std::vector<cv::Point2f> corners;
    if (!cv::findChessboardCornersSB(gray, board.pattern_size(), corners)) {
        why = "no checkerboard found";
        return {};
    }
    if (board.colours_fix_orientation()) {
        const double contrast = colour_contrast(gray, board, corners);
        if (std::abs(contrast) < min_contrast) {
            why = "square colours too faint to fix the orientation";
            return {};
        }
        if (contrast < 0) std::reverse(corners.begin(), corners.end());
    }
    return corners;
}
