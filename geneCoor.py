'''
generate coordinate(int) for software live according to the undistortion model
'''

import numpy as np
import pickle
import tifffile

def distort_model(params, x,y):
    fx,fy,cx,cy,k1, k2, k3, p1, p2 = params
    matrix=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    objpoints=np.concatenate((x[:,np.newaxis],y[:,np.newaxis],np.ones_like(y[:,np.newaxis])),axis=1)
    objpoints_rotated=np.matmul(objpoints, matrix)
    objpoints_projected = objpoints_rotated[:, :2] / (objpoints_rotated[:, 2:] + 1e-17)
    shift=objpoints_projected-np.array([cx,cy])

    x_shifted = shift[:,0]
    y_shifted = shift[:,1]
    r2 = x_shifted**2 + y_shifted**2
    x_distorted = x_shifted * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p1*x_shifted*y_shifted + p2*(r2 + 2*x_shifted**2) + cx
    y_distorted = y_shifted * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + p1*(r2 + 2*y_shifted**2) + 2*p2*x_shifted*y_shifted + cy
    return x_distorted, y_distorted

def undistort_coor(params):
    H, W = (10748,14304)
    gty,gtx = np.mgrid[:H, :W]
    gtxy = np.c_[gtx.ravel(), gty.ravel()]
    x_undistorted, y_undistorted = distort_model(params['inv_undistort'],(gtxy[:,0]-W//2)/100,(gtxy[:,1]-H//2)/100)
    x_undistorted=x_undistorted*100+W//2
    y_undistorted=y_undistorted*100+H//2
    return x_undistorted,y_undistorted

with open("./undistort_params_dict_points_240529.pkl",'rb') as file:
    params=pickle.load(file)

H,W,centerX,centerY = 10748, 14304, 7152, 5373
crop_H, crop_W = 10005, 14025
grid_x = np.arange(centerX-crop_W//2+7, centerX+crop_W//2+1, 15, dtype=np.int16)
grid_y = np.arange(centerY-crop_H//2+7, centerY+crop_H//2+1, 15, dtype=np.int16)
gtx,gty = np.meshgrid(grid_x, grid_y)
print(gtx.shape)
gtxy = np.c_[gtx.ravel(), gty.ravel()]
x_undistorted, y_undistorted = distort_model(params['inv_undistort'],(gtxy[:,0]-W//2)/100,(gtxy[:,1]-H//2)/100)
x_undistorted=np.round(x_undistorted*100+W//2).astype(np.int16)
y_undistorted=np.round(y_undistorted*100+H//2).astype(np.int16)
coor = np.c_[y_undistorted.ravel(), x_undistorted.ravel()]
with open('./coordinates.txt', 'w') as file:
    for row in coor:
        file.write(f'{row[0]},{row[1]}\n')

image = tifffile.imread(r"Z:\2_Data\LWJ\RUSH3D\Data\White\20240529\White_panorama_20X_C2\capture\C2\White_panorama_20X_S1_C2_2.tiff")
tmp = image
cv = tmp[coor[:,0],coor[:,1]].reshape(667,935)
tifffile.imwrite('./cv.tif', cv)


# image = np.roll(image[:9],4,axis=0)
# order = [6,5,4,7,8,3,0,1,2]
# image = image[order]
# cv = np.zeros((9,667,935),dtype=np.uint16)
# for i in range(9):
#     tmp = image[i]
#     cv[i] = tmp[coor[:,0],coor[:,1]].reshape(667,935)

# cv = cv.reshape(3,3,667,935).transpose(2,0,3,1).reshape(3*667,3*935)
# tifffile.imwrite('./cv.tif', cv)
