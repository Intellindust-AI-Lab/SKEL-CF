
import os
from typing import List
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2

from vis.color import ColorPalette



os.environ['PYOPENGL_PLATFORM'] = 'egl'

class Renderer(object):

    def __init__(self, focal_length=600, img_w=512, img_h=512,
                 same_mesh_color=False):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.same_mesh_color = same_mesh_color


    def render_front_view(self, verts, color_name, faces, bg_img_rgb=None, bg_color=(1., 1., 1., 1.)):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)
        # for every person in the scene
        for n in range(num_people):

            mesh = trimesh.Trimesh(verts[n], faces)
            mesh.apply_transform(rot)

            if self.same_mesh_color:
                mesh_color = ColorPalette.presets_float[color_name]
            else:
                mesh_color = colorsys.hsv_to_rgb((float(n) / num_people), 0.6, 1.0)

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(*mesh_color, 1.0))
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            scene.add(mesh, 'mesh')
        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        # transfrom to bgr
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        if bg_img_rgb is None:
            return color_bgr
        else:
            valid_mask = (depth_map > 0)[:,:,None]
            # bg_img_rgb[mask] = color_rgb[mask]
            # return bg_img_rgb
            # visible_weight = 0.9
            output_img = (
                color_bgr[:, :, :3] * valid_mask 
                + bg_img_rgb * (1-valid_mask) 
            )
            
            output_img = np.clip(output_img, 0, 255).astype(np.uint8)
            return output_img

    def render_side_view(self, verts, color_name, faces, bg_img_rgb=None, bg_color=(1., 1., 1., 1.), mesh_alpha: float = 0.6, enable_shadows: bool = True):
        mesh_base_color = ColorPalette.presets_float[color_name]
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='BLEND' if mesh_alpha < 1.0 else 'OPAQUE',
            baseColorFactor=(*mesh_base_color, float(np.clip(mesh_alpha, 0.0, 1.0))))
        
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                cx=self.camera_center[0], cy=self.camera_center[1])
        
        # bg_color
        if len(bg_color) == 3:
            bg_color = (*bg_color, 0.0)
        
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.array([0.3, 0.3, 0.3]))
        scene.add(camera, pose=np.eye(4))

        # Lights
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0]))
        scene.add(light, pose=trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0]))

        if isinstance(verts, np.ndarray) and verts.ndim == 2:
            people_verts = [verts]
        else:
            people_verts = list(verts)
        
        for n, vtx in enumerate(people_verts):
            mesh = trimesh.Trimesh(vtx, faces, process=False)
            
            # 应用旋转：先绕Y轴90度（侧视），再绕X轴180度（翻转）
            rot_y = trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
            mesh.apply_transform(rot_y)
            rot_x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot_x)
            
            # 向+Z方向平移，确保在相机视野内
            trans = trimesh.transformations.translation_matrix([0.0, 0.0, 1.2])
            mesh.apply_transform(trans)
            
            mesh_py = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh_py, f'mesh_side_{n}')
        
        color_rgba, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        
        color_rgb = color_rgba[:, :, :3]
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        
        if bg_img_rgb is None:
            return color_bgr

        alpha = (color_rgba[:, :, 3:4].astype(np.float32)) / 255.0
        fg = color_bgr.astype(np.float32)
        bg = bg_img_rgb.astype(np.float32)
        comp = fg * alpha + bg * (1.0 - alpha)
        return comp.astype(np.uint8)


    def render_top_view(self, verts, color_name, faces):
        centroid = verts.mean(axis=(0, 1))   # (3,)
        centroid[:2] = 0

        aroundx = cv2.Rodrigues(np.array([np.radians(90.), 0, 0]))[0][np.newaxis, ...]  # (1,3,3)
        pred_vert_arr_top = np.matmul((verts - centroid), aroundx) + centroid
        
        pred_vert_arr_top[:,:,2] += 1
        top_view = self.render_front_view(pred_vert_arr_top, color_name, faces)
        return top_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()