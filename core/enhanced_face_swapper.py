import torch
import mediapipe as mp
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from PIL import Image
from absl import logging 
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import dlib
from scipy.spatial import Delaunay
import kornia

class EnhancedFaceSwapper:
    """Improved face swapping between source and target images with special eye region handling"""
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        
    def swap_faces(self, source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
        """Enhanced face swapping with special handling for eye regions to prevent double-eye effect"""
        print("Starting enhanced face swap with eye region handling...")
        
        # 1. Validate and prepare images
        if source_img is None or target_img is None:
            print("Error: One or both input images are None")
            return target_img if target_img is not None else np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Ensure 3-channel images
        if len(source_img.shape) != 3 or source_img.shape[2] != 3:
            source_img = cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR) if len(source_img.shape) == 2 else cv2.cvtColor(source_img, cv2.COLOR_BGRA2BGR)
        if len(target_img.shape) != 3 or target_img.shape[2] != 3:
            target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR) if len(target_img.shape) == 2 else cv2.cvtColor(target_img, cv2.COLOR_BGRA2BGR)
            
        # 2. Detect faces with Dlib
        source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        
        source_faces = self.face_detector(source_gray)
        target_faces = self.face_detector(target_gray)
        
        if len(source_faces) == 0 or len(target_faces) == 0:
            print(f"Found {len(source_faces)} face(s) in source image and {len(target_faces)} face(s) in target image")
            return target_img
            
        print(f"Found {len(source_faces)} face(s) in source image")
        print(f"Found {len(target_faces)} face(s) in target image")
        
        # 3. Get facial landmarks using Dlib (68 points)
        source_shape = self.shape_predictor(source_gray, source_faces[0])
        target_shape = self.shape_predictor(target_gray, target_faces[0])
        
        source_landmarks = np.array([[p.x, p.y] for p in source_shape.parts()], dtype=np.float32)
        target_landmarks = np.array([[p.x, p.y] for p in target_shape.parts()], dtype=np.float32)
        
        # 4. Extract eye landmarks
        # Left eye indices: 36-41, Right eye indices: 42-47
        left_eye_src = source_landmarks[36:42]
        right_eye_src = source_landmarks[42:48]
        left_eye_tgt = target_landmarks[36:42]
        right_eye_tgt = target_landmarks[42:48]
        
        # Create debug visualization
        debug_img = target_img.copy()
        # Draw all landmarks
        for point in target_landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)
        
        # Highlight eye landmarks in a different color
        for eye_points in [left_eye_tgt, right_eye_tgt]:
            for point in eye_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(debug_img, (x, y), 3, (255, 0, 0), -1)
                
        cv2.imwrite('debug_eye_landmarks.jpg', debug_img)
        
        # 5. Calculate convex hull for face masking
        source_hull = cv2.convexHull(source_landmarks.astype(np.int32))
        target_hull = cv2.convexHull(target_landmarks.astype(np.int32))
        
        # 6. Create tight eye region masks to handle eye regions specially
        left_eye_mask_src = np.zeros(source_img.shape[:2], dtype=np.uint8)
        right_eye_mask_src = np.zeros(source_img.shape[:2], dtype=np.uint8)
        left_eye_mask_tgt = np.zeros(target_img.shape[:2], dtype=np.uint8)
        right_eye_mask_tgt = np.zeros(target_img.shape[:2], dtype=np.uint8)
        
        # Draw eye regions with slight padding for better coverage
        cv2.fillConvexPoly(left_eye_mask_src, self._expand_eye_region(left_eye_src), 255)
        cv2.fillConvexPoly(right_eye_mask_src, self._expand_eye_region(right_eye_src), 255)
        cv2.fillConvexPoly(left_eye_mask_tgt, self._expand_eye_region(left_eye_tgt), 255)
        cv2.fillConvexPoly(right_eye_mask_tgt, self._expand_eye_region(right_eye_tgt), 255)
        
        # Combine eye masks
        eyes_mask_src = cv2.bitwise_or(left_eye_mask_src, right_eye_mask_src)
        eyes_mask_tgt = cv2.bitwise_or(left_eye_mask_tgt, right_eye_mask_tgt)
        
        # Slightly dilate for better coverage
        eyes_mask_src = cv2.dilate(eyes_mask_src, None, iterations=2)
        eyes_mask_tgt = cv2.dilate(eyes_mask_tgt, None, iterations=2)
        
        # Lip indices: 48-68
        lip_landmarks = target_landmarks[48:68]

        # Create a lip mask similar to eye masks
        lip_mask_tgt = np.zeros(target_img.shape[:2], dtype=np.uint8)
        lip_hull = cv2.convexHull(lip_landmarks.astype(np.int32))
        cv2.fillConvexPoly(lip_mask_tgt, lip_hull, 255)
        lip_mask_tgt = cv2.dilate(lip_mask_tgt, None, iterations=2)

        # Save eye masks for debugging
        cv2.imwrite('debug_eyes_mask.jpg', eyes_mask_tgt)
        # Save for debugging
        cv2.imwrite('debug_lip_mask.jpg', lip_mask_tgt)
                
        # 7. Create face masks
        source_mask = np.zeros(source_img.shape[:2], dtype=np.uint8)
        target_mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
   
        cv2.fillConvexPoly(source_mask, source_hull, 255)
        cv2.fillConvexPoly(target_mask, target_hull, 255)
        
        # 8. Create better feathered edge for the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        target_mask_dilated = cv2.dilate(target_mask, kernel, iterations=1)
        target_mask_feathered = cv2.GaussianBlur(target_mask_dilated, (25, 25), 15)
        
        # 9. Delaunay triangulation
        rect = cv2.boundingRect(target_landmarks.astype(np.int32))
        subdiv = cv2.Subdiv2D(rect)
        
        for point in target_landmarks:
            subdiv.insert(tuple(map(float, point)))
            
        triangles = subdiv.getTriangleList()
        triangle_indices = []
        
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            
            # Convert triangle points to landmark indices
            idx1 = self._find_closest_landmark(target_landmarks, pt1)
            idx2 = self._find_closest_landmark(target_landmarks, pt2)
            idx3 = self._find_closest_landmark(target_landmarks, pt3)
            
            if idx1 is not None and idx2 is not None and idx3 is not None:
                triangle_indices.append([idx1, idx2, idx3])
                
        print(f"Generated {len(triangle_indices)} triangles using Delaunay triangulation")
        
        # 10. Initialize result image
        result_img = np.copy(target_img)
        
        # 11. Transform triangles from source to target, with special handling for eye regions
        for triangle in triangle_indices:
            # Check if this triangle is part of an eye region
            is_eye_triangle = False
            for idx in triangle:
                # Eye region landmarks: 36-47
                if 36 <= idx <= 47:
                    is_eye_triangle = True
                    break
            
            # Get triangle points
            source_tri = source_landmarks[triangle]
            target_tri = target_landmarks[triangle]
            
            # Apply special warping for eye triangles to prevent double-eye effect
            if is_eye_triangle:
                self._warp_triangle_special(source_img, result_img, source_tri, target_tri, 
                                           eyes_mask_src, eyes_mask_tgt, higher_precision=True)
            else:
                self._warp_triangle(source_img, result_img, source_tri, target_tri)
        
        # 12. Apply color correction before blending
        color_corrected = self._match_color_tones(result_img, target_img, target_mask_feathered)
        
        # 13. Apply modified seamless cloning for natural blending
        # Calculate center point avoiding the eyes
        # Use face landmarks excluding eyes for center calculation
        face_points = np.vstack([
            target_landmarks[0:36],  # Jaw and eyebrows
            target_landmarks[48:]    # Mouth and nose
        ])
        center = np.mean(face_points, axis=0).astype(np.int32)
        
        # Create a copy of the target image
        output = np.copy(target_img)
        
        # Create a mask that excludes eyes to avoid double-eye effect
        blend_mask = cv2.bitwise_and(
            target_mask_feathered, 
            cv2.bitwise_not(cv2.bitwise_or(eyes_mask_tgt, lip_mask_tgt))
        )

        # Check if we need special handling for drastically different skin tones
        color_corrected = self._handle_skin_tone_difference(color_corrected, target_img, target_mask_feathered)

        # Apply seamless cloning with the modified mask
        try:
            output = cv2.seamlessClone(
                color_corrected, 
                output, 
                blend_mask, 
                tuple(center), 
                cv2.MIXED_CLONE
            )
        except cv2.error as e:
            print(f"Seamless cloning failed: {e}")
            # Fall back to alpha blending
            alpha = target_mask_feathered.astype(float) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            output = (color_corrected * alpha + target_img * (1 - alpha)).astype(np.uint8)
        
        # 14. Special handling for eye regions - use the source eye regions directly
        # Extract eye regions from source and warp them to target eye positions
        for src_eye, tgt_eye, eye_mask in [
            (left_eye_src, left_eye_tgt, left_eye_mask_tgt),
            (right_eye_src, right_eye_tgt, right_eye_mask_tgt)
        ]:
            # Calculate transformation matrix to align eyes
            src_eye_center = np.mean(src_eye, axis=0)
            tgt_eye_center = np.mean(tgt_eye, axis=0)
            
            # Get eye width and height
            src_eye_width = np.max(src_eye[:, 0]) - np.min(src_eye[:, 0])
            tgt_eye_width = np.max(tgt_eye[:, 0]) - np.min(tgt_eye[:, 0])
            
            # Calculate scale factor
            scale = tgt_eye_width / src_eye_width if src_eye_width > 0 else 1.0
            
            # Get rotation angle
            src_angle = np.arctan2(src_eye[3][1] - src_eye[0][1], 
                                  src_eye[3][0] - src_eye[0][0])
            tgt_angle = np.arctan2(tgt_eye[3][1] - tgt_eye[0][1], 
                                  tgt_eye[3][0] - tgt_eye[0][0])
            angle = tgt_angle - src_angle
            
            # Create transformation matrix
            rotation_matrix = cv2.getRotationMatrix2D(
                tuple(src_eye_center), angle * 180 / np.pi, scale)
            
            # Add translation
            rotation_matrix[0, 2] += tgt_eye_center[0] - src_eye_center[0]
            rotation_matrix[1, 2] += tgt_eye_center[1] - src_eye_center[1]
            
            # Extract eye region with some margin
            margin = 5  # Adjust this value as needed
            x_min, y_min = np.min(src_eye, axis=0) - margin
            x_max, y_max = np.max(src_eye, axis=0) + margin
            
            # Ensure coordinates are within image bounds
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(source_img.shape[1], int(x_max))
            y_max = min(source_img.shape[0], int(y_max))
            
            if x_max <= x_min or y_max <= y_min:
                continue  # Skip if eye region is invalid
                
            # Extract eye region
            eye_region = source_img[y_min:y_max, x_min:x_max]
            
            # Warp eye region
            warped_eye = cv2.warpAffine(
                source_img, rotation_matrix, 
                (target_img.shape[1], target_img.shape[0]),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Apply eye mask for smooth blending
            eye_mask_blurred = cv2.GaussianBlur(eye_mask, (5, 5), 2)
            eye_blend_mask = np.expand_dims(eye_mask_blurred.astype(float) / 255.0, axis=2)
            
            # Blend warped eye into output
            output = (warped_eye * eye_blend_mask + 
                     output * (1 - eye_blend_mask)).astype(np.uint8)
        
        # Create final debug visualization
        debug_final = output.copy()
        for point in target_landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(debug_final, (x, y), 2, (0, 255, 0), -1)
        
        # Highlight eye landmarks
        for eye_points in [left_eye_tgt, right_eye_tgt]:
            for point in eye_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(debug_final, (x, y), 3, (0, 0, 255), -1)
                
        cv2.imwrite('debug_enhanced_final.jpg', debug_final)
        
        print("Enhanced face swap with eye region handling completed successfully")
        return output
        
    def _expand_eye_region(self, eye_points, padding=2):
        """Expand eye region by adding padding to ensure full coverage"""
        # Calculate eye center
        eye_center = np.mean(eye_points, axis=0)
        
        # Expand points outward from center
        expanded_points = []
        for point in eye_points:
            # Vector from center to point
            vector = point - eye_center
            # Normalize vector
            length = np.sqrt(np.sum(vector ** 2))
            if length > 0:
                unit_vector = vector / length
                # Expand point outward
                expanded_point = point + unit_vector * padding
                expanded_points.append(expanded_point)
            else:
                expanded_points.append(point)
                
        return np.array(expanded_points, dtype=np.int32)
        
    def _find_closest_landmark(self, landmarks, point):
        """Find the closest landmark index to a given point"""
        distances = np.sqrt(np.sum((landmarks - point) ** 2, axis=1))
        min_dist_idx = np.argmin(distances)
        
        # Use a threshold to ensure accurate matching
        if distances[min_dist_idx] < 5:
            return min_dist_idx
        return None
        
    def _warp_triangle(self, src_img, dst_img, src_tri, dst_tri):
        """Standard triangle warping for non-eye regions"""
        # Get bounding rectangle for destination triangle
        rect = cv2.boundingRect(dst_tri.astype(np.int32))
        (x, y, w, h) = rect
        
        # Check if rectangle is within image bounds
        if x < 0 or y < 0 or x + w > dst_img.shape[1] or y + h > dst_img.shape[0]:
            return
            
        # Offset triangles by the rectangular region
        dst_tri_cropped = np.array([
            [dst_tri[0][0] - x, dst_tri[0][1] - y],
            [dst_tri[1][0] - x, dst_tri[1][1] - y],
            [dst_tri[2][0] - x, dst_tri[2][1] - y]
        ], dtype=np.float32)
        
        src_tri_cropped = np.array([
            [src_tri[0][0], src_tri[0][1]],
            [src_tri[1][0], src_tri[1][1]],
            [src_tri[2][0], src_tri[2][1]]
        ], dtype=np.float32)
        
        # Create mask for destination triangle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri_cropped.astype(np.int32), 255)
        
        # Warp source triangle to match destination
        warp_mat = cv2.getAffineTransform(src_tri_cropped, dst_tri_cropped)
        
        # Warp the source image
        warped = cv2.warpAffine(
            src_img, 
            warp_mat, 
            (w, h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Apply mask to keep only the triangle region
        warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
        
        # Create inverse mask for the original content
        mask_inv = cv2.bitwise_not(mask)
        original_cropped = cv2.bitwise_and(dst_img[y:y+h, x:x+w], dst_img[y:y+h, x:x+w], mask=mask_inv)
        
        # Combine original and warped triangles
        dst_img[y:y+h, x:x+w] = cv2.add(original_cropped, warped_masked)
        
    def _warp_triangle_special(self, src_img, dst_img, src_tri, dst_tri, 
                              src_eye_mask, dst_eye_mask, higher_precision=False):
        """Special triangle warping for eye regions with precise alignment"""
        # Get bounding rectangle for destination triangle
        rect = cv2.boundingRect(dst_tri.astype(np.int32))
        (x, y, w, h) = rect
        
        # Check if rectangle is within image bounds
        if x < 0 or y < 0 or x + w > dst_img.shape[1] or y + h > dst_img.shape[0]:
            return
            
        # Offset triangles by the rectangular region
        dst_tri_cropped = np.array([
            [dst_tri[0][0] - x, dst_tri[0][1] - y],
            [dst_tri[1][0] - x, dst_tri[1][1] - y],
            [dst_tri[2][0] - x, dst_tri[2][1] - y]
        ], dtype=np.float32)
        
        src_tri_cropped = np.array([
            [src_tri[0][0], src_tri[0][1]],
            [src_tri[1][0], src_tri[1][1]],
            [src_tri[2][0], src_tri[2][1]]
        ], dtype=np.float32)
        
        # Create mask for destination triangle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri_cropped.astype(np.int32), 255)
        
        # For eye regions, check if this triangle is in the eye mask
        eye_mask_roi = dst_eye_mask[y:y+h, x:x+w] if (y+h <= dst_eye_mask.shape[0] and x+w <= dst_eye_mask.shape[1]) else np.zeros((h, w), dtype=np.uint8)
        
        # For triangles in eye regions, use higher precision warping
        if higher_precision:
            flags = cv2.INTER_CUBIC
        else:
            flags = cv2.INTER_LINEAR
        
        # Warp source triangle to match destination
        try:
            warp_mat = cv2.getAffineTransform(src_tri_cropped, dst_tri_cropped)
            
            # Warp the source image
            warped = cv2.warpAffine(
                src_img, 
                warp_mat, 
                (w, h), 
                flags=flags, 
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Apply mask to keep only the triangle region
            warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
            
            # Create inverse mask for the original content
            mask_inv = cv2.bitwise_not(mask)
            original_cropped = cv2.bitwise_and(dst_img[y:y+h, x:x+w], dst_img[y:y+h, x:x+w], mask=mask_inv)
            
            # Check if this area overlaps with an eye region
            if np.sum(eye_mask_roi) > 0:
                # Use alpha blending for smoother transition in eye regions
                alpha_mask = mask.astype(float) / 255.0
                alpha_mask = np.expand_dims(alpha_mask, axis=2)
                
                # Modify alpha for eye regions - reduce visibility of source
                if higher_precision:
                    # Reduce the alpha value for smoother blending in eye regions
                    eye_mask_roi_norm = eye_mask_roi.astype(float) / 255.0
                    eye_mask_roi_norm = np.expand_dims(eye_mask_roi_norm, axis=2)
                    
                    # Scale alpha to prevent double-eyes
                    alpha_mask = alpha_mask * (1.0 - eye_mask_roi_norm * 0.3)
                
                # Blend using modified alpha
                blended = warped_masked * alpha_mask + dst_img[y:y+h, x:x+w] * (1.0 - alpha_mask)
                dst_img[y:y+h, x:x+w] = blended.astype(np.uint8)
            else:
                # For non-eye regions, use standard addition
                dst_img[y:y+h, x:x+w] = cv2.add(original_cropped, warped_masked)
                
        except cv2.error:
            # Skip triangles that cause errors
            pass

    def _adjust_lighting(self, src_face, tgt_face, mask):
        """Correct lighting differences between source and target"""
        # Convert to grayscale for luminance analysis
        src_gray = cv2.cvtColor(src_face, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(tgt_face, cv2.COLOR_BGR2GRAY)
        
        # Get mean luminance in masked regions
        src_mean = cv2.mean(src_gray, mask=mask)[0]
        tgt_mean = cv2.mean(tgt_gray, mask=mask)[0]
        
        # Calculate adjustment factor
        alpha = tgt_mean / src_mean if src_mean > 0 else 1.0
        
        # Apply lighting correction
        result = cv2.addWeighted(src_face, alpha, np.zeros_like(src_face), 0, 0)
        return result

    # Add a skin smoothing step
    def _blend_skin_texture(self, src_face, tgt_face, mask):
        """Blend skin textures for more natural results"""
        # Apply bilateral filter to preserve edges but smooth skin
        smoothed = cv2.bilateralFilter(src_face, 9, 75, 75)
        
        # Blend original with smoothed
        mask_3ch = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2) / 255.0
        alpha = 0.3  # Control amount of smoothing
        blended = src_face * (1-alpha) + smoothed * alpha
        
        return blended.astype(np.uint8)

    # Improve color correction by enhancing the _match_color_tones function:
    def _match_color_tones(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Enhanced color matching between source and target images"""
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        
        # Split channels
        ls, as_, bs = cv2.split(source_lab)
        lt, at, bt = cv2.split(target_lab)
        
        # Calculate mean and std for each channel
        channels = [(ls, lt), (as_, at), (bs, bt)]
        
        for src, tgt in channels:
            src_mean, src_std = cv2.meanStdDev(src, mask=mask)
            tgt_mean, tgt_std = cv2.meanStdDev(tgt, mask=mask)
            
            if src_std[0][0] > 0:
                src[:] = (((src - src_mean[0][0]) * (tgt_std[0][0]/src_std[0][0])) + 
                        tgt_mean[0][0])
        
        # Merge channels
        corrected_lab = cv2.merge([ls, as_, bs])
        corrected = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        # Create distance-based weight map
        y, x = np.indices(mask.shape)
        face_indices = np.where(mask > 0)
        
        if len(face_indices[0]) > 0:
            center_y = np.mean(face_indices[0])
            center_x = np.mean(face_indices[1])
            
            # Calculate distance from center
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            max_dist = np.max(dist[mask > 0])
            dist_normalized = dist / max_dist if max_dist > 0 else dist
            
            # Create weight map
            weight_map = np.ones_like(mask, dtype=np.float32)
            weight_map[mask > 0] = 1.0 - 0.3 * dist_normalized[mask > 0]
            
            # Apply Gaussian blur to weight map
            weight_map = cv2.GaussianBlur(weight_map, (25, 25), 15)
            
            # Create proper 3-channel masks for blending
            blend_mask = cv2.GaussianBlur(mask.astype(np.float32), (25, 25), 15) / 255.0
            blend_mask = np.expand_dims(blend_mask, axis=2)
            weight_map = np.expand_dims(weight_map, axis=2)
            
            # Final blending
            final_mask = blend_mask * weight_map
            result = (corrected * final_mask + source * (1.0 - final_mask))
        else:
            # Fallback to simple alpha blending
            mask_3ch = np.expand_dims(mask.astype(float) / 255.0, axis=2)
            result = corrected * mask_3ch + source * (1.0 - mask_3ch)
        
        return result.astype(np.uint8)

    def _normalize_skin_tone(self, face_img, face_mask):
        """Normalize skin tones to reduce oily appearance with drastic skin tone differences"""
        # Create a copy for processing
        result = face_img.copy()
        
        # Convert to YCrCb color space which is better for skin tone processing
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to luminance
        # This helps normalize lighting while preserving local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_normalized = clahe.apply(y)
        
        # Normalize color channels (Cr, Cb) to reduce excessive saturation in skin tones
        # Calculate mean and std in masked region
        cr_mean, cr_std = cv2.meanStdDev(cr, mask=face_mask)
        cb_mean, cb_std = cv2.meanStdDev(cb, mask=face_mask)
        
        # Define target values (these are typical values for natural skin)
        cr_target_mean, cr_target_std = 153, 12
        cb_target_mean, cb_target_std = 110, 12
        
        # Normalize channels
        if cr_std[0][0] > 0:
            cr = ((cr - cr_mean[0][0]) * (cr_target_std / cr_std[0][0])) + cr_target_mean
        if cb_std[0][0] > 0:
            cb = ((cb - cb_mean[0][0]) * (cb_target_std / cb_std[0][0])) + cb_target_mean
        
        # Merge channels and convert back
        normalized_ycrcb = cv2.merge([y_normalized, cr, cb])
        normalized = cv2.cvtColor(normalized_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # Blend normalized result with original only in masked region
        mask_float = face_mask.astype(float) / 255.0
        mask_3ch = np.repeat(np.expand_dims(mask_float, axis=2), 3, axis=2)
        
        # Apply a bilateral filter to preserve edges while smoothing
        normalized = cv2.bilateralFilter(normalized, 9, 75, 75)
        
        # Final blend with gradual alpha to avoid hard edges
        mask_3ch_blurred = cv2.GaussianBlur(mask_3ch, (31, 31), 15)
        result = normalized * mask_3ch_blurred + face_img * (1 - mask_3ch_blurred)
        
        return result.astype(np.uint8)

    def _preserve_skin_texture(self, source_face, target_face, blended_face, face_mask):
        """Preserve facial texture details when blending different skin tones"""
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
        blended_lab = cv2.cvtColor(blended_face, cv2.COLOR_BGR2LAB)
        
        # Extract L channels
        l_source, _, _ = cv2.split(source_lab)
        l_target, _, _ = cv2.split(target_lab)
        l_blended, a_blended, b_blended = cv2.split(blended_lab)
        
        # Calculate detail layer from target using high-pass filter
        blurred = cv2.GaussianBlur(l_target, (31, 31), 0)
        detail_target = l_target.astype(np.float32) - blurred.astype(np.float32)
        
        # Apply detail to blended image
        # We only want to transfer high-frequency details (texture), not color
        detail_strength = 0.4  # Adjust as needed (0.3-0.5 is usually good)
        l_result = l_blended.astype(np.float32) + detail_target * detail_strength
        
        # Clip values to valid range
        l_result = np.clip(l_result, 0, 255).astype(np.uint8)
        
        # Merge channels
        result_lab = cv2.merge([l_result, a_blended, b_blended])
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        # Apply only in masked area with feathering
        mask_feathered = cv2.GaussianBlur(face_mask, (21, 21), 11)
        mask_3ch = np.repeat(np.expand_dims(mask_feathered, axis=2), 3, axis=2) / 255.0
        
        final = blended_face * (1 - mask_3ch) + result * mask_3ch
        
        return final.astype(np.uint8)

    def _handle_skin_tone_difference(self, source_img, target_img, face_mask):
        """Apply specialized processing when source and target skin tones are very different"""
        # Check if skin tones are significantly different
        source_face_region = cv2.bitwise_and(source_img, source_img, mask=face_mask)
        target_face_region = cv2.bitwise_and(target_img, target_img, mask=face_mask)
        
        # Convert to HSV for better skin tone comparison
        source_hsv = cv2.cvtColor(source_face_region, cv2.COLOR_BGR2HSV)
        target_hsv = cv2.cvtColor(target_face_region, cv2.COLOR_BGR2HSV)
        
        # Get mean values of V (value/brightness) channel
        _, _, source_v = cv2.split(source_hsv)
        _, _, target_v = cv2.split(target_hsv)
        
        source_v_mean = cv2.mean(source_v, mask=face_mask)[0]
        target_v_mean = cv2.mean(target_v, mask=face_mask)[0]
        
        # Calculate difference
        v_diff = abs(source_v_mean - target_v_mean)
        
        # If significant difference detected
        if v_diff > 30:  # Threshold can be adjusted
            print(f"Significant skin tone difference detected: {v_diff:.2f}. Applying specialized processing.")
            
            # 1. First normalize the skin tone to reduce the oily appearance
            normalized = self._normalize_skin_tone(source_img, face_mask)
            
            # 2. Apply color matching
            color_matched = self._match_color_tones(normalized, target_img, face_mask)
            
            # 3. Preserve texture details
            result = self._preserve_skin_texture(source_img, target_img, color_matched, face_mask)
            
            # 4. Optional: Apply very subtle noise to avoid overly smooth appearance
            noise = np.zeros_like(result)
            cv2.randn(noise, 0, 3)  # Subtle noise
            result = cv2.add(result, noise, mask=face_mask)
            
            return result
        else:
            # Standard processing is fine
            return self._match_color_tones(source_img, target_img, face_mask)

    def _create_nose_mask(self, landmarks, shape):
        """Create specialized mask for the nose region with better transition"""
        # Get nose landmarks (27-35) and surrounding landmarks
        nose_points = landmarks[27:36]  # Main nose landmarks
        upper_lip_points = landmarks[48:55]  # Upper lip points for better transition
        
        # Create extended points to include area below nose
        extended_points = np.vstack([
            nose_points,
            upper_lip_points,
            landmarks[31:36]  # Add bottom nose points again for stronger weight
        ])
        
        # Create initial nose mask
        nose_mask = np.zeros(shape, dtype=np.uint8)
        
        # Create hull for the extended region
        nose_hull = cv2.convexHull(extended_points.astype(np.int32))
        cv2.fillConvexPoly(nose_mask, nose_hull, 255)
        
        # Create gradual falloff mask for smoother transition
        falloff_mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(falloff_mask, nose_hull, 255)
        
        # Apply progressive blurring for smoother transition
        kernel_sizes = [3, 5, 7]  # Multiple blur passes with increasing size
        for kernel_size in kernel_sizes:
            falloff_mask = cv2.GaussianBlur(
                falloff_mask, 
                (kernel_size, kernel_size), 
                0
            )
        
        # Normalize the falloff mask
        falloff_mask = falloff_mask.astype(float) / 255.0
        
        # Create final mask with smooth transition
        final_mask = (nose_mask.astype(float) * falloff_mask).astype(np.uint8)
        
        return final_mask
    
    def _handle_lip_region(self, source_img, target_img, lip_landmarks, lip_mask):
        """Special handling for lip region to preserve lip color and texture"""
        # Create lip region hull
        lip_hull = cv2.convexHull(lip_landmarks.astype(np.int32))
        
        # Extract lip regions
        x, y, w, h = cv2.boundingRect(lip_hull)
        lip_roi = source_img[y:y+h, x:x+w]
        
        # Apply color preservation
        lip_mask_roi = lip_mask[y:y+h, x:x+w]
        preserved_lips = cv2.addWeighted(
            lip_roi, 0.7,  # 70% original lip color
            target_img[y:y+h, x:x+w], 0.3,  # 30% target lip color
            0
        )
        
        # Blend back with feathered edges
        mask_feathered = cv2.GaussianBlur(lip_mask_roi, (7, 7), 3)
        mask_3ch = np.repeat(np.expand_dims(mask_feathered, axis=2), 3, axis=2) / 255.0
        target_img[y:y+h, x:x+w] = (preserved_lips * mask_3ch + 
                                   target_img[y:y+h, x:x+w] * (1 - mask_3ch))
        
        return target_img

    def _feather_mask_edges(self, mask, feather_amount=15):
        """Create smooth feathered edges for any mask"""
        # Ensure feather amount is odd
        if feather_amount % 2 == 0:
            feather_amount += 1
            
        # Create gradual feathering
        mask_feathered = cv2.GaussianBlur(
            mask, 
            (feather_amount, feather_amount),
            feather_amount//4
        )
        
        # Create stronger feathering at edges
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)
        edge_feather = cv2.GaussianBlur(
            edge_mask,
            (feather_amount*2+1, feather_amount*2+1),
            feather_amount//2
        )
        
        # Combine masks
        final_mask = cv2.addWeighted(
            mask_feathered, 0.7,
            edge_feather, 0.3,
            0
        )
        
        return final_mask