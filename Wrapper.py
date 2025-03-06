import os
import cv2 
import argparse
import numpy as np
from tqdm import tqdm
import scipy.optimize

def read_images(path):
    print(f"Reading images from {path} for Calibration")
    images = []
    if not os.path.exists(path):
        print(f"Error: Directory '{path}' not found.")
        exit(1)   
    for image_name in sorted(os.listdir(path)):
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)
            print(f"Loaded image[{len(images) - 1}] from {image_path}")
        else:
            print(f"Could not read {image_path}")

    return images

def world_coord(length, width, square_side):
    world_x, world_y = np.meshgrid(range(length), range(width))
    world_xyz = np.hstack((world_x.reshape(-1, 1), world_y.reshape(-1, 1)))
    world_xyz = world_xyz * square_side
    
    return world_xyz

def checkerboard_coord(images, pattern):

    corners = []

    out = os.path.join("Output", "Detected_Chessboard_Corners")
    os.makedirs(out, exist_ok=True)

    for i,image in enumerate(images):
        image_copy = image.copy()
        ret, corner = cv2.findChessboardCorners(image, pattern, None)
        if ret:
            corner = np.squeeze(corner)
            cv2.drawChessboardCorners(image_copy, pattern, corner, ret)
            cv2.imwrite(os.path.join(out, str(i+1)+'.jpg'), image_copy)
        else:
            continue

        # corners.append([image, corner])
        corners.append(corner)
    
    # Format: [[img_name, corners], [img_name, corners], ...]
    # Corners: shape ((length * width), 1, 2) 
    return corners 

def find_homography(corners, world_xyz):
    H = []
    for i in range(len(corners)):
        h, _ = cv2.findHomography(world_xyz, corners[i])
        h = h / h[2, 2]
        H.append(h)
    return H

def find_v_ij(h, i, j):

    v_ij =  np.array([
            h[0, i] * h[0, j],
            h[0, i] * h[1, j] + h[1, i] * h[0, j],
            h[1, i] * h[1, j],
            h[2, i] * h[0, j] + h[0, i] * h[2, j],
            h[2, i] * h[1, j] + h[1, i] * h[2, j],
            h[2, i] * h[2, j]
    ])
    return v_ij.T

def find_intrinsic_K(H_list):

    V = []
    for H in H_list:
        H = H / H[2, 2]  # Normalize H to ensure H[2,2] = 1
        V.append(find_v_ij(H, 0, 1))
        V.append(find_v_ij(H, 0, 0) - find_v_ij(H, 1, 1))

    V = np.array(V)
    # print(V.shape)

    # Solve for b using SVD
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]

    # Construct the intrinsic matrix K from b
    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lambda_ / B11)
    beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_
    u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_

    K = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    return K

def find_extrinsics(H_list, K):

    extrinsics = []
    K_inv = np.linalg.inv(K)

    for H in H_list:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        lambda_ = 1 / np.linalg.norm(np.dot(K_inv, h1))

        r1 = lambda_ * np.dot(K_inv, h1)
        r2 = lambda_ * np.dot(K_inv, h2)
        r3 = np.cross(r1, r2)  # Compute r3 as the cross product of r1 and r2

        R = np.column_stack((r1, r2, r3))
        t = lambda_ * np.dot(K_inv, h3)
        extrinsics.append(np.column_stack((R, t)))

    return extrinsics
    
def point_projection(X, K, R, t, k):

    X_homogeneous = np.hstack((X, np.ones((X.shape[0], 1))))
    X_camera = (R @ X_homogeneous.T).T + t.T
    x_projected = X_camera[:, :2] / X_camera[:, 2][:, np.newaxis]
    
    r2 = np.sum(x_projected**2, axis=1)
    x_distorted = x_projected * (1 + k[0] * r2 + k[1] * r2**2)[:, np.newaxis]
    
    x_final = (K @ np.hstack((x_distorted, np.ones((x_distorted.shape[0], 1)))).T).T

    return x_final[:, :2]

def reprojection_error_function(parameters, world_xyz, corners, num_imgs, r=False):

    fx = parameters[0]
    fy = parameters[1]
    gamma = parameters[2]
    u0 = parameters[3]
    v0 = parameters[4]

    K = np.array([
        [fx, gamma, u0],
        [0, fy, v0],
        [0,  0,  1]
    ])

    Rt_parameters = parameters[5:-2].reshape(num_imgs, 6)
    k = parameters[-2:]

    corner_reproj = []
    error = []

    for i in range(num_imgs):
        r_vec = Rt_parameters[i][:3]  # Rotation vector
        t_vec = Rt_parameters[i][3:].reshape(3, 1)  # Translation vector
        R, _ = cv2.Rodrigues(r_vec)  # Convert rotation vector to matrix

        x_projected = point_projection(world_xyz, K, R, t_vec, k)

        error.append((corners[i] - x_projected).flatten())
        corner_reproj.append(x_projected)

    if r:
        return np.hstack(error), corner_reproj

    return np.hstack(error)

def optimize_function(K, Rt, k, world_xyz, corners):
    num_imgs = len(Rt)

    Rt_flattened = np.array([np.hstack((cv2.Rodrigues(Rt[:, :3])[0].flatten(), Rt[:, -1]))for Rt in Rt]).flatten()
    initial_parameters = np.hstack((K.flatten(), Rt_flattened, k))

    result = scipy.optimize.least_squares(fun=reprojection_error_function,x0=initial_parameters,args=(world_xyz, corners, num_imgs),method='lm')

    return result.x

def visualize_reprojected_points(images, K, k, reprojected_points, detected_corners):

    out = os.path.join("Output", "Reprojected_Corners")
    os.makedirs(out, exist_ok=True)

    for i, (image, reprojected, detected) in enumerate(zip(images, reprojected_points, detected_corners)):
        image_copy = image.copy()
        image_copy = cv2.undistort(image_copy, K, k)
        reprojected = reprojected.reshape(-1, 2)
        detected = detected.reshape(-1, 2)


        # Draw reprojected points in red
        for point in reprojected:
            cv2.circle(image_copy, (int(point[0]),int(point[1])), 7, (0, 0, 255), -1)
        
        # Draw detected corners in green
        for point in detected:
            cv2.circle(image_copy, (int(point[0]),int(point[1])), 4, (0, 255, 0), -1)

        # Save the visualization
        output_path = os.path.join(out, f"reprojected_{i+1}.jpg")
        cv2.imwrite(output_path, image_copy)



def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument("--Path", type=str, default="Calibration_Imgs/", help="Path to the main directory containing image sets")
    Args = Parser.parse_args()
    path = Args.Path

    images = read_images(path)

    pattern = (9, 6)
    square_side = 21.5 # in mm

    world_xyz = world_coord(pattern[0], pattern[1], square_side)
    corners = checkerboard_coord(images, pattern)

    H = find_homography(corners, world_xyz)
    K = find_intrinsic_K(H)
    print('\nCamera Calibration Matrix before Optimizing:', K)
    Rt = find_extrinsics(H, K)

    k = np.array([0.0, 0.0])  # Initial distortion coefficients
    fx = K[0][0]
    fy = K[1][1]
    gamma = K[0][1]
    u0 = K[0][2]
    v0 = K[1][2]

    K_init = np.array([fx, fy, gamma, u0, v0])

    Rt_flattened = np.array([np.hstack((cv2.Rodrigues(Rt[:, :3])[0].flatten(), Rt[:, -1]))for Rt in Rt]).flatten()
    initial_parameters = np.hstack((K_init.flatten(), Rt_flattened, k))


    new_parameters = optimize_function(K_init, Rt, k, world_xyz, corners)

    new_fx = new_parameters[0]
    new_fy = new_parameters[1]
    new_gamma = new_parameters[2]
    new_u0 = new_parameters[3]
    new_v0 = new_parameters[4]

    new_K = np.array([  [new_fx, new_gamma, new_u0],
                        [0.0,          new_fy,     new_v0],
                        [0.0,          0.0,              1.0]])

    # new_Rt = new_parameters[5:-2].reshape(len(corners), 6)
    new_k = new_parameters[-2:]
    new_k = np.array([new_k[0], new_k[1], 0, 0, 0])
    print('\nDistortion Coefficients after Calibration are', new_k[0], 'and', new_k[1])
    print('\nCamera Calibration Matrix after Optimizing:', new_K)


    reproj_error, reproj_corners = reprojection_error_function(new_parameters, world_xyz, corners, len(images), r=True)
    visualize_reprojected_points(images, new_K, new_k, reproj_corners, corners)
    print("Reprojection Error after Optimization")
    reproj_error = reproj_error.reshape(-1, pattern[0]*pattern[1]*2)
    reproj_error_per_image = np.mean(np.abs(reproj_error), axis=1)

    for i, error in enumerate(reproj_error_per_image):
        print(f"For Image {i+1}, Reprojection Error is {error:.3f}")

    mean_reproj_error = np.mean(reproj_error_per_image)
    print(f"\nThe Mean Reprojection Error Over All Images is {mean_reproj_error:.3f}")


    reproj_error, _ = reprojection_error_function(initial_parameters, world_xyz, corners, len(images), r=True)
    print("\n\nReprojection Error before Optimization")
    reproj_error = reproj_error.reshape(-1, pattern[0]*pattern[1]*2)
    reproj_error_per_image = np.mean(np.abs(reproj_error), axis=1)

    for i, error in enumerate(reproj_error_per_image):
        print(f"For Image {i+1}, Reprojection Error is {error:.3f}")

    mean_reproj_error = np.mean(reproj_error_per_image)
    print(f"\nThe Mean Reprojection Error Over All Images is {mean_reproj_error:.3f}")





if __name__ == "__main__":
    main()