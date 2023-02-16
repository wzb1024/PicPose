import math
import numpy as np

def angle(v1, v2):
  dx1 = v1[2] - v1[0]
  dy1 = v1[3] - v1[1]
  dx2 = v2[2] - v2[0]
  dy2 = v2[3] - v2[1]
  angle1 = math.atan2(dy1, dx1)
  angle1 = angle1 * 180/math.pi
  # print(angle1)
  angle2 = math.atan2(dy2, dx2)
  angle2 = angle2 * 180/math.pi
  # print(angle2)
  if angle1*angle2 >= 0:
    included_angle = abs(angle1-angle2)
  else:
    included_angle = abs(angle1) + abs(angle2)
  return included_angle


def get_angles(points):
    points=np.array(points)
    link=list()
    for i in range(4):
        l=list()
        l.append(points[0][:2])
        l.append(points[i*4+1:i*4+5,:2])
        link.append(l)
        
    angle_points=list()
    for it in link:
        l=list()
        for i in range(3):
            l.append(np.append(it[0],it[1][i]))
            l.append(np.append(it[1][i+1],it[1][i]))
        l.append(np.append(it[1][0],it[1][1]))
        l.append(np.append(it[1][2],it[1][1]))
        l.append(np.append(it[1][1],it[1][2]))
        l.append(np.append(it[1][3],it[1][2]))
        angle_points.append(l)

    angles=list()
    for it in angle_points:
        l=list()
        for i in range(5):
            if 0 in it[i*2+1] or 0 in it[i*2]:
                l.append(-1)
                continue
            l.append(angle(it[i*2],it[i*2+1]))
        angles.append(l)
    
    return angles