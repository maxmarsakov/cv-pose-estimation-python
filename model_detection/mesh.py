"""
mesh parser
"""
from tkinter import VERTICAL
import numpy as np


class Mesh:

    def __init__(self, vertices, triangles):
        
        self.vertices_ = vertices
        self.triangles_ = triangles

    def loadRoofMesh(self, alpha=1.):
        """
        from the mesh, form the pyramidic roof, 
        including 4 vertices for which Z != 0
        The height of the roof will be (alpha + 1) * height of the box
        returns (vertices, mesh)
        """
        # get the height of the box
        height = np.max(self.vertices_[:,2])

        # get the vertices forming the base of the roof 
        pyramid_base = self.vertices_[:,2] > 0
        pyramid_base_vertices = self.vertices_[pyramid_base][:4,:]
        
        # form additional roof shpitz  vertice
        roof_shpitz = np.array([
            (np.max(pyramid_base_vertices[:,0]) + np.min(pyramid_base_vertices[:,0])) // 2,
            (np.max(pyramid_base_vertices[:,1]) + np.min(pyramid_base_vertices[:,1])) // 2,
            height * (alpha + 1)])
            
        # form a mesh - sort the indices
        def ordinator(x):
            if x[0] == 0 and x[1] == 0:
                return 0
            if x[0] != 0 and x[1] == 0:
                return 1
            if x[0] != 0 and x[1] != 0:
                return 2
            return 3

        sorted_vertices = np.argsort(np.apply_along_axis(ordinator, axis=1,arr=pyramid_base_vertices))
        pyramid_base_vertices = pyramid_base_vertices[sorted_vertices]
        #append shpitz to all vertices
        mesh = []
        for i in range(4):
            mesh.append( (4,i,(i+1)%4) )
        
        pyramid_base_vertices = np.vstack([pyramid_base_vertices, roof_shpitz])
        return mesh,pyramid_base_vertices

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