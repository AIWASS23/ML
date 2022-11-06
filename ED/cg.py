
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy as dp
import cv2
from random import randint

#IMPORTANTS
#1 - Refactorings
#1A - OOP implements will need for a reading and reduce code complexity
#1A - Objects :
        # Sphere{center,radius}
        # Poligon{points}
        # Triangle{points}

def compute_vector_w(origin):
    vector_w = []
    norm_origin = np.linalg.norm(origin);
    for i in origin:
        vector_w.append(i/norm_origin);
    return vector_w

def compute_vector_u(vector_w):
    vector_u = []
    vector_t = compute_vector_t(vector_w)
    cross_w_and_t = np.cross(vector_w,vector_t)
    norm_cross_w_and_t = np.linalg.norm(cross_w_and_t)
    for i in cross_w_and_t:
        vector_u.append (i / norm_cross_w_and_t );
    return vector_u

def compute_vector_v(vector_w,vector_u):
    vector_v = []
    cross_w_and_u = np.cross (vector_w, vector_u)
    for i in cross_w_and_u:
        vector_v.append (i);
    return vector_v

def compute_vector_t(vector_w):
    index_min_value = vector_w.index(min(np.abs(vector_w)))
    vector_temp = dp(vector_w)
    if(vector_temp[index_min_value] == 1 ):
        vector_temp[index_min_value] = 0
    else:
        vector_temp[index_min_value] = 1
    return vector_temp

def compute_vectors_wuv(origin):
    vector_w = compute_vector_w (origin)
    vector_u = compute_vector_u (vector_w)
    vector_v = compute_vector_v (vector_w, vector_u)
    return vector_w,vector_u,vector_v

def compute_escalar_ul(indexX,left,right,nx):
    ul = left + (right - left)*(indexX + 0.5)/nx
    return ul

def compute_escalar_vl(indexY,top,bottom,ny):
    vl = bottom + (top - bottom)*(indexY + 0.5)/ny
    return vl

def compute_directions(origin,distance,left,right,top,bottom,nx,ny):
    directions = []
    # vector_w, vector_u, vector_v = compute_vectors_wuv (origin)
    for i in range(nx):
        for j in range(ny):
            direction = compute_direction(i,j,origin,distance,left,right,top,bottom,nx,ny)
            directions.append(direction)
    return directions

def compute_direction(x,y,origin,distance,left,right,top,bottom,nx,ny):
    vector_w, vector_u, vector_v = compute_vectors_wuv (origin)
    ul = compute_escalar_ul(x, left, right, nx)
    vl = compute_escalar_vl(y, top, bottom, ny)
    return (np.dot(distance,vector_w) + np.dot(ul,vector_u) + np.dot(vl,vector_v))


def solve_quadratic_equation(A,B,C):
    delta = B**2 - 4 * A * C
    t1 = t2 = ""
    if delta >= 0:
        t1 = (-B - delta**0.5)/A
        t2 = (-B + delta**0.5)/A
    return delta,t1,t2


def compute_circle(origin,direction,center_circle,radius):
    eminusc = np.subtract(origin,center_circle)
    dot_dir_dir = np.dot(direction,direction)
    dot_dir_eminusc = np.dot(direction,eminusc)
    dot_eminusc_eminusc = np.dot(eminusc,eminusc) - pow(radius,2)
    return solve_quadratic_equation(dot_dir_dir,dot_dir_eminusc,dot_eminusc_eminusc)

def sphere_detection_in_oblique_case(origem=[5,5,5],distance=0.2,sphere_center=[0,0,0],radius=8,left=-10,right=10,bottom=-10,top=10,nx=200,ny=200):
    imagem = np.zeros ((nx, ny, 3), dtype=int)
    # cv2.imwrite ("sphere_detection.jpg", imagem)
    for x in range (0, imagem.shape[0]):
        for y in range (0, imagem.shape[1]):
            direction = (compute_direction (x, y, origem, -1 / distance, left, right, top, bottom, nx, ny))
            # <modify for a variable sphere quantity>
            # vetorize spheres objects -> iterate sphere array ->compute delta value
            delta, t1, t2 = compute_circle(origem, direction,sphere_center , radius)
            # <end of modify/>
            print ("passando =[{x}][{y}]".format (x=x, y=y), end="")
            print (" - delta = " + str (delta))
            if delta >= 0:
                imagem[x][y] = [0, 255, 255]
    text = 'Sphere Detection'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText (imagem, text, (5, ny - 5), font, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite ("sphere_detection.jpg", imagem)

def multiple_sphere_detection_in_oblique_case():
    print("not implemented function...")


def poligon_detection():
    print("not implemented function...")


def triangle_detection():
    print("not implemented function...")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy as dp
from src import raytrace_


origem = [1,1,1]
l = 10
r = 5
t = 10
b = 5
nx = 5
ny = 5
distance = 1

fig = plt.figure()
ax = fig.gca(projection='3d')

directions = raytrace_.compute_directions(origem, distance, l, r, t, b, nx, ny)
# ax.quiver(0, 0, 0, 1, 1, 1, normalize=True)
origem = np.dot(raytrace_.compute_vector_w(origem),-1)

for i in directions:
    ax.quiver(origem[0],origem[1],origem[2],i[2],i[1],i[0])
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy as dp
from src import raytrace_
from src import Sphere
# origem = [5,5,5]
# l = -10
# r = 10
# t = 10
# b = -10
# nx = 120
# ny = 120
# distance = 0.1
#
# raytrace_.sphere_detection_in_oblique_case(name="0.2")
# raytrace_.sphere_detection_in_oblique_case(distance=distance,name="0.1")



retaA = [[0, 0, 0.2], [0, 0.2,  0]]
retaB = [[0, 0, 0.2], [0, 0.2,0.1]]


fig = plt.figure()
ax = fig.gca(projection='3d')


ax.quiver(retaA[0][0], retaA[0][1], retaA[0][2], retaA[1][0], retaA[1][1], retaA[1][2])
ax.quiver(retaB[0][0], retaB[0][1], retaB[0][2], retaB[1][0], retaB[1][1], retaB[1][2])
# ax.quiver(reta[2][0], reta[2][1], reta[2][2], reta[0][0], reta[0][1], reta[0][2])
# ax.quiver(poligon[3][0],poligon[3][1],poligon[3][2],poligon[0][0],poligon[0][1],poligon[0][2])
plt.show()

import cv2
from config import config
import copy

class Image:
    def __init__(self, nameimage):
        self.image = None
        self.nameimage = nameimage

    def average_filter(self, kernel_size):
        radius = int(kernel_size / 2)
        filtered_image = copy.deepcopy(self.image)
        for colunm in range(self.image.shape[0]):
            for line in range(self.image.shape[1]):
                kernel = []
                self.fillkernel(colunm, line, kernel, radius)
                filtered_image[colunm][line] = int(sum(kernel) / kernel_size ** 2)
        self.write_text(filtered_image, "Average Applied")
        cv2.imwrite(config.PATH_OUTPUT_IMAGES +"AVERAGED_"+str(i) + self.nameimage+".jpg",
                     filtered_image)


    def median_filter(self, kernel_size,i):
        radius = int(kernel_size / 2)
        filtered_image = copy.deepcopy(self.image)
        for colunm in range(self.image.shape[0]):
            for line in range(self.image.shape[1]):
                kernel = []
                self.fillkernel(colunm, line, kernel, radius)
                filtered_image[colunm][line] = sorted(kernel)[int((len(kernel))/2)]
        self.write_text(filtered_image,"Median Filter Applied")
        cv2.imwrite(config.PATH_OUTPUT_IMAGES+"MEDIAN_IN_"+str(i)+self.nameimage+".jpg",
                    filtered_image)

    def contrast_redefinition(self):
        filtered_image = copy.deepcopy(self.image)
        histogram = [0]*256
        size = self.image.size
        for colunm in self.image:
            for line in colunm:
                histogram[line] += 1
        for _p in range(0, len(histogram)):
            histogram[_p] = histogram[_p] / self.image.size
        for _p in range(1, len(histogram)):
            histogram[_p] = histogram[_p] + histogram[_p - 1]
        for _p in range(0, len(histogram)):
            histogram[_p] = int(round((((len(histogram) - 1) * histogram[_p])), 0))
        for colunm in range(self.image.shape[0]):
            for line in range(self.image.shape[1]):
                filtered_image[colunm][line] = histogram[filtered_image[colunm][line]]

        self.write_text (filtered_image, "Equalization Applied")
        cv2.imwrite (config.PATH_OUTPUT_IMAGES + "EQUALIZATION_" + self.nameimage,
                     filtered_image)

    def limiarization(self, threshold):
        filtered_image = copy.deepcopy(self.image)
        for colunm in range(self.image.shape[0]):
            for line in range(self.image.shape[1]):
                if(self.image[colunm][line] > threshold):
                    filtered_image[colunm][line] = 255
                else:
                    filtered_image[colunm][line] = 0
        self.write_text(filtered_image, "Limiarization Applied")
        cv2.imwrite(config.PATH_OUTPUT_IMAGES + "LIMIARIZATION_" + self.nameimage,
                    filtered_image)


    def load_image(self, modo = 0):
        try:
            path_image = config.PATH_INPUT_IMAGES + self.nameimage
            self.image = cv2.imread (path_image, modo)
            if self.image is None:
                raise IOError("FILE {namefile} NOT FOUND".format(namefile=self.nameimage))
        except IOError as exc:
            print("ERROR : {args} ".format(args=exc.args))

    def show_image(self):
        try:
            if self.image is None:
                raise IOError ("FILE {namefile} NOT FOUND".format (namefile=self.nameimage))
            cv2.imshow(self.nameimage, self.image)
            cv2.waitKey(0)
            cv2.destroyAllwindows()
        except IOError as exc:
            print("ERROR : {args} ".format(args=exc.args))

    def fillkernel(self, colunm, line, kernel, radius):
        for l in range(colunm - radius, colunm + radius + 1):
            for j in range(line - radius, line + radius + 1):
                if (l + colunm < 0 or j + line < 0):
                    kernel.append(0)
                elif (l + radius > self.image.shape[0] or j + radius > self.image.shape[1]):
                    kernel.append(0)
                else:
                    kernel.append(self.image[l][j])

    def write_text(self,image,text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 25)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        cv2.putText(image, text,
                     bottomLeftCornerOfText,
                     font,
                     fontScale,
                     fontColor,
                     lineType)

    def setImage(self,np_array,pos_x,pos_y):
        for l in range(0,self.image.shape[0]):
            for c in range(0, self.image.shape[1]):
                np_array[l + int(pos_x / 2)][c + int(pos_y / 2)] = float(self.image[l][c])