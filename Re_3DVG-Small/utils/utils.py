import numpy as np
import open3d as o3d
def getRt(fragmentsNum,pose_matrix_path,start_ids=1):
    """
    Args:

        fragmentsNum: fragmentsNum: 取第n个rt（0，5，10，...）
        pose_matrix_path: pose文件的路径

    Returns:
        rt

    """
    with open(pose_matrix_path, mode="r") as r:
        lines = r.readlines()
        rt = []
        rt.append(list(map(float, lines[fragmentsNum * 5 + start_ids].strip().split())))
        rt.append(list(map(float, lines[fragmentsNum * 5 + start_ids+1].strip().split())))
        rt.append(list(map(float, lines[fragmentsNum * 5 + start_ids+2].strip().split())))
        rt.append(list(map(float, lines[fragmentsNum * 5 + start_ids+3].strip().split())))
        r.close()
    return np.array(rt)
def transform(rt, points):
    """

    Args:
        rt: 变换矩阵
        points: 预转换的点云坐标

    Returns:
        new_points: 转换后的点云坐标

    """
    points = np.asarray(points)
    R = rt[:, :3][:3]
    T = rt[:, 3][:3]
    points_rt = np.dot(R, points.transpose((1, 0)))
    new_points = points_rt.transpose((1, 0)) + T
    return new_points
def find_indexes(arr1, arr2):
    indexes = []
    for val in arr1:
        if val in arr2:
            indexes.append(np.where(arr2 == val)[0][0])
    return indexes
def get_obb(points,i):
    """

    Args:
        points: object点云的xyz坐标

    Returns:
        obb包围盒的中心坐标
        vertex_set: 8个顶点坐标

    """
    pcd = o3d.geometry.PointCloud()
    if len(points)<4:
        # points=points.repeat(4,axis=0)
        for i in range(4):
            points.extend(points)

    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200,
                                             std_ratio=0.2)
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=1000,
    #                                          std_ratio=0.1)
    try:
        pcd = pcd.select_by_index(ind)
        # 这句话偶尔会出现bug，如果出现bug就忽略掉这个region，然后返回None
        obb = pcd.get_oriented_bounding_box()
        [center_x, center_y, center_z] = obb.get_center()
        # obb包围盒的顶点
        vertex_set = np.asarray(obb.get_box_points())
        # print("obb包围盒的中心坐标为：\n", [center_x, center_y, center_z])
        # obb.color = (0, 1, 0)  # obb包围盒为绿色
        # o3d.visualization.draw_geometries([pcd, obb], window_name="OBB包围盒",
        #                                   width=1024, height=768,
        #                                   left=50, top=50,
        #                                   mesh_show_back_face=False)
        o3d.io.write_point_cloud('/home/light/文档/0731_kinect/ply/{}.ply'.format(i), pcd)
        return [center_x, center_y, center_z], vertex_set
    except:
        return None, None


def get_box3d_min_max(corner):
    """

    Args:
        corner: numpy array (8,3), assume up direction is Z (batch of N samples)

    Returns:
        an array for min and max coordinates of 3D bounding box IoU

    """

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


def box3d_iou(corners1, corners2):
    """

    Args:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z

    Returns:
        iou: 3D bounding box IoU

    """
    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou


def get_iou_obb(bb1, bb2):
    iou3d = box3d_iou(bb1, bb2)
    return iou3d

import os
def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "/" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件，就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)
