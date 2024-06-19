import numpy as np
from scipy.spatial.transform import Rotation

from .colmap_loader import CameraModel, Camera, BaseImage, Point3D, CAMERA_MODELS, CAMERA_MODEL_IDS, CAMERA_MODEL_NAMES, Image, rotmat2qvec, qvec2rotmat, read_next_bytes

def read_densecloud_extrinsics(path: str, scale=1.0, scale_depths=False):
    """
    Reads the extrinsics and associated image data from EuRoC pose files
    as ORB-SLAM3 provides them and converts the values to the COLMAP format for
    gaussian splatting to work properly.

    The format of the poses from ORB-SLAM3 are:
    ```
    image_id    t_wc(0)     t_wc(1)     t_wc(2)      q_wc(0)     q_wc(1)     q_wc(2)     q_wc(3)
    ```
    where the transformations are all given as representing the homogenous
    transform T_WC (camera to world) matrix. The quaternions are saved as
    (qx, qy, qz, qw) by ORB-SLAM3.

    COLMAP poses are saved as the T_CW (world to camera) transform and the
    quaternions are saved in a different order (qw, qx, qy, qz). This function
    converts everything to as how `read_extrinsics_text()` would return.
    """
    images = {}
    camera_id = 1 # we treat only monocular cases
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(float(elems[0])) # convert to float first to get rid of decimal points

                # Read in the quaternions and translation vector for T_WC (camera to world) transform
                qx, qy, qz, qw = tuple(map(float, elems[4:8]))
                qvec_wc = np.array([qx, qy, qz, qw])
                rot_WC = Rotation.from_quat(qvec_wc)
                rot_CW = rot_WC.inv()
                R_WC = rot_WC.as_matrix()
                R_CW = rot_CW.as_matrix()
                qvec_cw = rot_CW.as_quat()

                qvec = np.concatenate((qvec_cw[-1:], qvec_cw[:-1])) # colmap has the qw term first and wants q_cw (world to cam)

                # Need to get the translation vector of the inverse transform
                tvec = np.array(tuple(map(float, elems[1:4]))) # t_wc (cam to world, from orb-slam3)
                tvec = - R_CW @ tvec[:, None] # t_cw = - R_WC.T @ t_wc, where R_WC.T == R_CW
                tvec = tvec.squeeze() # rest of the code expects (3,) shape

                # If we do not scale depths we need to scale the poses
                if not scale_depths:
                    tvec *= scale # scaled to match the depth predictions and the pointcloud

                # Assuming images are prepended zeros until length 5
                image_name = f"{image_id:05}.png"

                # These are normally there in the COLMAP images.txt files but we don't need them
                xys = None
                point3D_ids = None

                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)

    return images

def read_densecloud_extrinsics_colmap(path: str, scale=1.0, raw_colmap_file=True, scale_depths=False):
    """
    Heavily based off of `read_extrinsics_text` from the original repo.
    The only addition is to use a scaling factor for the translation components
    since the poses and the pointcloud come from separate sources.
    """
    images = {}

    if not raw_colmap_file:
        print("Assuming the colmap output images.txt has been cleaned such that it only has a single line per image and no information about the projections of 3D points to each image.")

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5]))) # qw qx qy qz

                tvec = np.array(tuple(map(float, elems[5:8]))) # tx ty tz

                # If we do not scale depths we need to scale the poses
                if not scale_depths:
                    tvec *= scale # scaled to match the depth predictions and the pointcloud

                camera_id = int(elems[8])
                image_name = elems[9]

                if raw_colmap_file:
                    elems = fid.readline().split()
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                else:
                    xys = None
                    point3D_ids = None

                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_densecloud_extrinsics_colmap_binary(path: str, scale=1.0, scale_depths=False):
    """
    Heavily based off of `read_extrinsics_binary` from the original repo.
    The only addition is to use a scaling factor for the translation components
    since the poses and the pointcloud come from separate sources.
    """
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])

            tvec = np.array(binary_image_properties[5:8])

            if not scale_depths:
                tvec = tvec * scale

            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images



def read_densecloud_intrinsics(path: str):
    """
    Read the camera model information and crop/resize information of the
    original images.

    The format is expected to be as:
    ```
    # Camera list with one line of data per camera, inspired from COLMAP format
    #    CAMERA_ID, MODEL, TARGET_WIDTH, TARGET_HEIGHT, INTRINSICS[], CROP_BOX[], SCALE
    #
    # Intrinsics are saved as a list of [fx, fy, cx, cy] values, they correspond to the intrinsics of the cropped and resized images
    #
    # Crop box is used to crop the original sized images before resizing such that the aspect ratio of the target sizes is preserved given as [left, upper, right, lower] edges
    # a value of all -1's indicates that no cropping will be done
    #
    # Scale is the float value used to multiply with the translations to get the resulting pointcloud and to match the predicted depths
    #
    # Number of cameras: 1
    1 PINHOLE 1024 576 535.894435796231 535.894435796231 511.90361445783134 287.90361445783134 0 0 5312 2988 20.0
    ```
    """
    cameras = {}

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()

            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(float(elems[0])) # convert to float first to remove decimals
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2]) # TODO figure out whether resizing images after reading them in breaks things downstream
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:8])))

                crop_box = list(map(int, elems[8:12])) # TODO figure out whether this breaks things downstream
                if crop_box == [-1, -1, -1, -1]:
                    crop_box = None

                try:
                    scale = float(elems[12])
                except IndexError:
                    print(f"Scale value cannot be read from {path}, assuming a value of 1.0")
                    scale = 1.0

                cameras[camera_id]= Camera(id=camera_id,
                                           model=model,
                                           width=width,
                                           height=height,
                                           params=params)

    return cameras, crop_box, scale