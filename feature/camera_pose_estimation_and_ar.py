import numpy as np
import cv2 as cv


def select_img_from_video(video, pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # Open video
    video = cv.VideoCapture(video)
    assert video.isOpened()

    # Select image
    img_select = []
    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show images
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 150), cv.FONT_HERSHEY_DUPLEX, 5, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):
                complete, pts = cv.findChessboardCorners(img, pattern)
                cv.drawChessboardCorners(display, pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img)
            if key == 27:
                break

    cv.destroyAllWindows()
    return img_select


def calib_camera_from_chessboard(images, pattern, cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(pattern[1]) for c in range(pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * cellsize] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)


def create_3d_sphere(center, radius, num_points=100):
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    sphere_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    return sphere_points.astype(np.float32)


def ar_object_visualization(video_file, board_pattern, board_cellsize, board_criteria, K, dist_coeff):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file

    # Prepare 3D points on a chessboard
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    # Sphere settings
    sphere_center = np.array([[4.5 * board_cellsize, 3 * board_cellsize, -0.5 * board_cellsize]])  # Center in chessboard coordinates

    # Run pose estimation
    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
            break

        # Estimate the camera pose
        success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
        if success:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

            # Project the sphere center point and draw it on the image
            sphere_3d_points = create_3d_sphere(sphere_center.flatten(), radius=board_cellsize)
            projected_sphere_points, _ = cv.projectPoints(sphere_3d_points, rvec, tvec, K, dist_coeff)
            for point in projected_sphere_points:
                cv.circle(img, tuple(point[0].astype(int)), radius=2, color=(0, 0, 255), thickness=-1)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 100), cv.FONT_HERSHEY_DUPLEX, 3 , (0, 255, 0))

        # Show the image and process the key event
        cv.imshow('Pose Estimation (Chessboard)', img)
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break

    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    video_file = '../sample/IPhone12Pro_4k_30fps.MOV'
    board_pattern = (10, 7)
    board_cellsize = 0.025
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'No selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    ar_object_visualization(video_file, board_pattern, board_cellsize, board_criteria, K, dist_coeff)
