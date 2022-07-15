
"""
mesh parser
"""
import numpy as np

class Mesh:

    def __init__(self, vertices, triangles):
        
        self.vertices_ = vertices
        self.triangles_ = triangles

    @staticmethod
    def loadMesh(filename: str):
        """
        reads PLY mesh file
        """
        vertices = []
        triangles = []
        vert_count = 0
        triangle_count=0
        with open(filename, "r") as fp:
            lines = fp.readlines()
            start_vertices=False
            start_faces=False
            max_vert = 0
            max_triangle = 0
            for line in lines:
                line_elems = line.split()
                if line_elems[0] == "element" and line_elems[1] == "face":
                    max_triangle=int(line_elems[2])
                elif line_elems[0] == "element" and line_elems[1] == "vertex":
                    max_vert=int(line_elems[2])
                elif line_elems[0] == "end_header":
                    start_vertices=True
                    continue
                elif start_vertices:
                    vertices.append( [float(v) for v in line_elems] )
                    vert_count += 1
                    if vert_count ==max_vert:
                        start_vertices=False
                        start_faces=True
                        continue
                elif start_faces:
                    triangles.append( [int(v) for v in line_elems][1:] )
                    triangle_count += 1
                    if triangle_count == max_triangle:
                        # done
                        break

        return Mesh(np.array(vertices), triangles)