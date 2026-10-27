#include "pose_estimator_node.h"

#include "charuco_board.h"
#include "corner_factor.h"

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>


PoseEstimatorNode::PoseEstimatorNode() 
:
    rclcpp::Node("pose_estimator"), 
    distortion_coeffs_({0.0, 0.0, 0.0, 0.0, 0.0})
{
    std::string board_type = this->declare_parameter<std::string>("board_type", "charuco");
    squares_x_ = this->declare_parameter<int>("squares_x");
    squares_y_ = this->declare_parameter<int>("squares_y");
    square_size_ = this->declare_parameter<double>("square_size");
    // Frame of the published pose; empty uses the image's frame_id (the camera optical frame)
    frame_id_ = this->declare_parameter<std::string>("frame_id", "");
    // Frames whose corner reprojection RMS exceeds this are rejected (bad detection)
    max_rms_px_ = this->declare_parameter<double>("max_rms_px", 3.0);

    if (board_type == "checkerboard") {
        board_type_ = BoardType::CHECKERBOARD;
        checkerboard_ = Checkerboard{squares_x_, squares_y_, square_size_};
        // Gray levels between white and black squares needed to trust the orientation
        min_colour_contrast_ = this->declare_parameter<double>("min_colour_contrast", 20.0);
        if (!checkerboard_->colours_fix_orientation()) {
            RCLCPP_WARN(get_logger(),
                "A %dx%d checkerboard looks the same turned 180 degrees: its pose is ambiguous. "
                "Use a board with squares_x + squares_y odd.", squares_x_, squares_y_);
        }
        RCLCPP_INFO(get_logger(), "Checkerboard %dx%d squares of %.4f m", squares_x_, squares_y_,
            square_size_);
    } else if (board_type == "charuco") {
        board_type_ = BoardType::CHARUCO;
        // Init ChArUco board from params
        std::string aruco_dict_name = this->declare_parameter<std::string>("aruco_dictionary");
        board_ = generate_charuco_board(aruco_dict_name, squares_x_, squares_y_, square_size_);

        // Save the board to file for debugging, etc
        cv::Mat image;
        int pixels_per_square = 300;
        cv::Size size(pixels_per_square * squares_x_, pixels_per_square * squares_y_);
        board_->draw(size, image, 100, 1);

        std::ostringstream ss;
        ss << "target_board_"
            << aruco_dict_name << "_"
            << squares_x_ << "x" << squares_y_ << ".png";

        std::string filename = ss.str();
        cv::imwrite(filename, image);
        RCLCPP_INFO(get_logger(), 
            "Saved target ChArUco board image to file: %s\n"
            "Ensure that your physical board matches this image!", 
            filename.c_str());
    } else {
        throw std::runtime_error("board_type must be 'charuco' or 'checkerboard', got '" + board_type + "'");
    }

    // Init pixel noise model from params
    double pixel_noise_sigma = this->declare_parameter<double>("pixel_noise_sigma");
    pixel_noise_ = gtsam::noiseModel::Isotropic::Sigma(2, pixel_noise_sigma);

    // Subscribers from topic name params
    std::string image_rect_topic = this->declare_parameter<std::string>("image_rect_topic");
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(image_rect_topic, 1, 
        std::bind(&PoseEstimatorNode::image_callback, this, std::placeholders::_1));
    
    std::string camera_info_topic = this->declare_parameter<std::string>("camera_info_topic");
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(camera_info_topic, 1, 
        std::bind(&PoseEstimatorNode::camera_info_callback, this, std::placeholders::_1));
    
    // Publishers
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/pose_estimator/pose", 1);
    image_pub_ = create_publisher<sensor_msgs::msg::Image>("/pose_estimator/annotated_image", 1);
    // Per accepted frame: [sum of squared corner reprojection errors (px^2), number of corners],
    // for calibrating pixel_noise_sigma (pooled sigma^2 = sum / (2 N - 6) over frames)
    residual_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/pose_estimator/residual", 10);

}


gtsam::Pose3 cv_to_gtsam(cv::Vec3d& rvec, cv::Vec3d& tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    gtsam::Matrix3 rot_mat;
    cv::cv2eigen(R, rot_mat);
    
    gtsam::Pose3 pose = gtsam::Pose3(
        gtsam::Rot3(rot_mat), 
        gtsam::Point3(tvec[0], tvec[1], tvec[2]));

    return pose;
}


Pose3Gaussian PoseEstimatorNode::optimize_board_pose(
    const std::vector<cv::Point3f>& board_points,
    const std::vector<cv::Point2f>& image_points,
    const gtsam::Pose3& pose_init)
{
    using namespace gtsam;
    
    // Build factor graph for optimization
    NonlinearFactorGraph graph;
    Key pose_key = gtsam::Symbol('T', 0);

    // Each pair of 2D/3D points constitutes a factor, i.e. an error to be minimized
    for (int i = 0; i < image_points.size(); i++) {
        Point3 point3 = Point3(board_points[i].x, board_points[i].y, board_points[i].z);
        Point2 point2 = Point2(image_points[i].x, image_points[i].y);

        graph.add(CornerFactor(pose_key, point3, point2, *camera_matrix_gtsam_, pixel_noise_));
    }

    // Init values object, needs initial guess for pose
    Values values;
    values.insert(pose_key, pose_init);

    // Do the optimization
    LevenbergMarquardtOptimizer opimizer(graph, values);
    values = opimizer.optimize();
    Marginals marginals(graph, values);

    // Extract solution from values/marginals
    Pose3Gaussian result;
    result.mean = values.at<Pose3>(pose_key);
    result.covariance = marginals.marginalCovariance(pose_key);
    // graph.error is half the whitened squared error; undo the whitening for pixels^2
    const double sigma = pixel_noise_->sigmas()(0);
    result.residual_sq = 2.0 * graph.error(values) * sigma * sigma;
    result.num_points = static_cast<int>(image_points.size());

    return result;
}


bool check_points_colinear(const std::vector<cv::Point2f>& pts) {
    cv::Mat data(pts.size(), 2, CV_64F);
    for (size_t i = 0; i < pts.size(); ++i) {
        data.at<double>(i, 0) = pts[i].x;
        data.at<double>(i, 1) = pts[i].y;
    }

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    double ratio = pca.eigenvalues.at<double>(1) /
                   pca.eigenvalues.at<double>(0);

    return ratio < 1e-2;  // threshold
}


std::optional<Pose3Gaussian> PoseEstimatorNode::process_frame(const cv::Mat& frame, cv::Mat& annotated) {
    frame.copyTo(annotated);
    if (board_type_ == BoardType::CHECKERBOARD) {
        return process_checkerboard_frame(frame, annotated);
    }
    return process_charuco_frame(frame, annotated);
}


std::optional<Pose3Gaussian> PoseEstimatorNode::process_checkerboard_frame(const cv::Mat& frame, cv::Mat& annotated) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::string why;
    auto corners = detect_checkerboard(gray, *checkerboard_, min_colour_contrast_, why);
    if (!corners) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Checkerboard: %s", why.c_str());
        return {};
    }
    cv::drawChessboardCorners(annotated, checkerboard_->pattern_size(), *corners, true);
    cv::circle(annotated, (*corners)[0], 8, cv::Scalar(0, 0, 255), 2);   // corner (0, 0)

    // IPPE: the planar PnP solution, as the initial guess for the refinement
    const std::vector<cv::Point3f> points_3d = checkerboard_->corners();
    cv::Vec3d rvec, tvec;
    if (!cv::solvePnP(points_3d, *corners, *camera_matrix_cv_, distortion_coeffs_, rvec, tvec,
                      false, cv::SOLVEPNP_IPPE)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Checkerboard: solvePnP failed");
        return {};
    }

    try {
        Pose3Gaussian pose = optimize_board_pose(points_3d, *corners, cv_to_gtsam(rvec, tvec));
        // The board z axis points into the printed face. Facing the camera, it points away from
        // it; otherwise the corners were mirrored (the board seen from behind).
        if (pose.mean.rotation().matrix()(2, 2) <= 0.0) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "Checkerboard: mirrored corner order (board facing away); rejected");
            return {};
        }
        draw_board_pose(pose.mean, annotated);
        return pose;
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
        RCLCPP_ERROR(get_logger(), "Optimization failed: %s", e.what());
        return {};
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Unexpected error during optimization: %s", e.what());
        return {};
    }
}


std::optional<Pose3Gaussian> PoseEstimatorNode::process_charuco_frame(const cv::Mat& frame, cv::Mat& annotated) {

    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    cv::aruco::detectMarkers(annotated, board_->dictionary, marker_corners, marker_ids, params_);
    cv::aruco::drawDetectedMarkers(annotated, marker_corners, {});

    if (marker_ids.size() == 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "No ArUco markers detected in the image"); 
        return {};
    }
        
    // Then get the charuco corners, i.e. the 2D detected points for pose estimation
    std::vector<cv::Point2f> points_2d;
    std::vector<int> charuco_ids;
    cv::aruco::interpolateCornersCharuco(
        marker_corners, 
        marker_ids,
        annotated, 
        board_,
        points_2d,
        charuco_ids,
        cv::Mat(), 
        cv::Mat()
    );

    cv::Scalar color = cv::Scalar(255, 0, 0);
    cv::aruco::drawDetectedCornersCharuco(annotated, points_2d, {}, color);

    // DLT requires at least 6 corners detected on the board
    int num_corners = points_2d.size();
    if (num_corners < 6) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "Not enough corners detected in the image: %d < 4", num_corners); 
        return {};
    }

    if (check_points_colinear(points_2d)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "Detected corners are not distributed well, cannot solvePnP"); 
        return {};
    }

    cv::Vec3d rvec;
    cv::Vec3d tvec;
    bool valid = cv::aruco::estimatePoseCharucoBoard(
        points_2d,
        charuco_ids,
        board_,
        *camera_matrix_cv_,
        distortion_coeffs_,
        rvec,
        tvec
    );

    if (!valid) {
        return {};
    }

    // Return if solution wasnt valid
    if (!valid) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "Invalid initial solvePnP solution");  
        return {};
    }

    // Now use gtsam optimizer to locally refine and get uncertainty
    gtsam::Pose3 pose_init = cv_to_gtsam(rvec, tvec);

    // This may throw an exception if optimization fails, catch and return empty optional
    try {
        // Extract 3D points in board frame from charuco corner ids
        std::vector<cv::Point3f> points_3d;
        points_3d.reserve(charuco_ids.size());
        for (size_t i = 0; i < charuco_ids.size(); ++i)
            points_3d.push_back(board_->chessboardCorners[charuco_ids[i]]);

        Pose3Gaussian pose = optimize_board_pose(points_3d, points_2d, pose_init);
        draw_board_pose(pose.mean, annotated);
        
        return pose;
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
        RCLCPP_ERROR(get_logger(), "Optimization failed: %s", e.what());
        return {};
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Unexpected error during optimization: %s", e.what());
        return {};
    }
}


void gtsam_to_cv(const gtsam::Pose3& pose, cv::Mat& R_out, cv::Vec3d& t_out) {
    cv::eigen2cv(pose.rotation().matrix(), R_out);
    gtsam::Point3 t = pose.translation();
    t_out = cv::Vec3d(t.x(), t.y(), t.z());
}


void PoseEstimatorNode::draw_board_pose(const gtsam::Pose3& pose, cv::Mat& image) const {
    cv::Mat R;
    cv::Vec3d t;
    gtsam_to_cv(pose, R, t);

    // Axis length: square length times min of num squares in x/y
    float axis_length = std::min(squares_x_, squares_y_) * square_size_;

    cv::drawFrameAxes(image, *camera_matrix_cv_, distortion_coeffs_, R, t, axis_length);
}


void PoseEstimatorNode::camera_info_callback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
    // If we already have camera info, ignore msg
    if (camera_matrix_cv_) {
        return;
    }

    camera_matrix_cv_ = cv::Mat(cv::Size(3, 3), CV_64F);

    for (size_t row = 0; row < 3; row++) {
        for (size_t col = 0; col < 3; col++) {
            // P is flattened 3x4 so P[row, col] = P[4*row + col]
            camera_matrix_cv_->at<double>(row, col) = msg->p.at(4 * row + col);
        }
    }

    camera_matrix_gtsam_ = gtsam::Cal3_S2(
        camera_matrix_cv_->at<double>(0,0), // fx
        camera_matrix_cv_->at<double>(1,1), // fy
        camera_matrix_cv_->at<double>(0,1), // skew
        camera_matrix_cv_->at<double>(0,2), // cx
        camera_matrix_cv_->at<double>(1,2)  // cy
    );

    RCLCPP_INFO(get_logger(), "Camera info received, ready to estimate board poses");
}


void PoseEstimatorNode::log_results(const Pose3Gaussian& pose, int elapsed_ms) const {

    const auto& t = pose.mean.translation();
    const auto& R = pose.mean.rotation().matrix();  // 3x3 rotation matrix

    double rx_std = std::sqrt(pose.covariance(0,0));
    double ry_std = std::sqrt(pose.covariance(1,1));
    double rz_std = std::sqrt(pose.covariance(2,2));

    double tx_std = std::sqrt(pose.covariance(3,3));
    double ty_std = std::sqrt(pose.covariance(4,4));
    double tz_std = std::sqrt(pose.covariance(5,5));

    RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        500.0,
        "\n===== Pose Estimate (%d ms elapsed)=====\n"
        "Position (mean, std) [m]:\n"
        "    x: %9.5f  ,  %9.5f\n"
        "    y: %9.5f  ,  %9.5f\n"
        "    z: %9.5f  ,  %9.5f\n"
        "Rotation matrix (mean):\n"
        "    [ %6.3f, %6.3f, %6.3f ]\n"
        "    [ %6.3f, %6.3f, %6.3f ]\n"
        "    [ %6.3f, %6.3f, %6.3f ]\n"
        "Rotation std dev [rad]:\n"
        "    roll : %6.3f\n"
        "    pitch: %6.3f\n"
        "    yaw  : %6.3f\n"
        "=========================",
        elapsed_ms,
        t.x(), tx_std,
        t.y(), ty_std,
        t.z(), tz_std,
        R(0,0), R(0,1), R(0,2),
        R(1,0), R(1,1), R(1,2),
        R(2,0), R(2,1), R(2,2),
        rx_std, ry_std, rz_std
    );
}


void PoseEstimatorNode::image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg) {
    // If we don't have camera info yet, igore msg
    if (!camera_matrix_cv_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "Waiting for camera_info before processing images");
        return;
    }

    // Convert using CV bridge
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
        return;
    }

    // Do the pose estimation
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat annotated;
    std::optional<Pose3Gaussian> pose = process_frame(cv_ptr->image, annotated);
    auto end = std::chrono::high_resolution_clock::now();
    int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Reject bad fits: the corners do not match the board well enough to trust the pose
    if (pose) {
        const double rms = std::sqrt(pose->residual_sq / pose->num_points);
        if (rms > max_rms_px_) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "Corner reprojection RMS %.2f px > max_rms_px %.2f; rejected", rms, max_rms_px_);
            pose.reset();
        } else {
            pooled_residual_sq_ += pose->residual_sq;
            pooled_dof_ += 2 * pose->num_points - 6;
            const double sigma_pos = 1e3 * std::sqrt(pose->covariance.bottomRightCorner<3, 3>().trace());
            cv::putText(annotated,
                cv::format("rms %.2f px  pooled sigma %.2f px  pos +-%.2f mm", rms,
                           std::sqrt(pooled_residual_sq_ / pooled_dof_), sigma_pos),
                cv::Point(20, annotated.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
    }

    // Publish annotated image no matter what
    cv_bridge::CvImage out;
    out.header = msg->header;
    out.encoding = "bgr8";
    out.image = annotated;

    image_pub_->publish(*out.toImageMsg());

    // Dont publish if the pose is bad
    if (!pose) {
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,   // ms
            "Board pose estimation failed"
        );
        return;
    }

    // Log/publish pose
    log_results(*pose, elapsed_ms);

    std_msgs::msg::Float64MultiArray residual;
    residual.data = {pose->residual_sq, static_cast<double>(pose->num_points)};
    residual_pub_->publish(residual);

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = msg->header.stamp;
    pose_msg.header.frame_id = frame_id_.empty() ? msg->header.frame_id : frame_id_;
    
    auto t = pose->mean.translation();
    pose_msg.pose.pose.position.x = t.x();
    pose_msg.pose.pose.position.y = t.y();
    pose_msg.pose.pose.position.z = t.z();

    gtsam::Quaternion q = pose->mean.rotation().toQuaternion();
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            pose_msg.pose.covariance[r*6 + c] = pose->covariance(r, c);
    
    pose_pub_->publish(pose_msg);
}