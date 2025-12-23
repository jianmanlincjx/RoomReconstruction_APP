import os
import json
import io
import base64
import time
import math
import re
import asyncio
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 

# Matplotlib 设置 (无头模式)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# 3D 绘图相关引用
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource, to_rgba
from shapely.geometry import Polygon, Point

# Swift 模型相关引用
from swift.llm import PtEngine, InferRequest, RequestConfig

# ==============================================================================
# 1. 全局配置与常量 (Config)
# ==============================================================================
MODEL_PATH = '/dataz/JM/checkpoint_tem/v0.0.8_unorder/v0-20250904-193049/Qwen2.5-VL-7B-Instruct'

class Config:
    # --- 字体与文本 (改为英文通用字体，彻底解决乱码) ---
    FONT_FAMILY = 'sans-serif' 
    MAIN_TITLE_FONT_SIZE, SUB_TITLE_FONT_SIZE = 28, 24
    AXIS_LABEL_FONT_SIZE, LEGEND_FONT_SIZE, ANNOTATION_FONT_SIZE = 20, 22, 18
    
    # --- 2D 绘图参数 ---
    WALL_COLOR, DOOR_COLOR, WINDOW_COLOR = 'black', 'royalblue', 'skyblue'
    OPENING_MARKER, OPENING_MARKER_SIZE = 'o', 80
    LINE_WIDTH_WALL, LINE_WIDTH_OPENING = 4.0, 8.0
    FIGURE_SIZE_2D, FIGURE_DPI = (16, 16), 120
    LABEL_OFFSET = 3.5

    # --- 3D 绘图参数 ---
    FIGURE_SIZE_3D = (12, 12)
    FONT_SIZE_3D_MAIN_TITLE, FONT_SIZE_3D_SUB_TITLE, FONT_SIZE_3D_DIM = 28, 22, 14
    
    # 物理尺寸
    WALL_HEIGHT = 2.7
    WALL_THICKNESS = 0.25
    COORDINATE_SYSTEM_MAX = 100.0
    TILE_SIZE_3D = 10
    
    # 门窗高度
    DEFAULT_DOOR_Z, DEFAULT_DOOR_H = 0.0, 2.1
    DEFAULT_WINDOW_Z, DEFAULT_WINDOW_H = 0.9, 1.5
    
    # 3D 颜色
    COLOR_WALL_3D = '#EAE0D5'
    COLOR_DOOR_PANEL_3D = '#8B4513'
    COLOR_WINDOW_PANEL_3D = '#ADD8E6'
    COLOR_FLOOR_3D = '#F5F5F5'
    COLOR_FLOOR_GRID_3D = '#D3D3D3'
    COLOR_TEXT_WALL_3D = '#5D4037'
    COLOR_TEXT_DOOR_3D = '#BF360C'
    COLOR_TEXT_WINDOW_3D = '#0D47A1'

config = Config()

# ==============================================================================
# 2. 全局变量
# ==============================================================================
model_engine = None
inference_lock = asyncio.Lock()

# ==============================================================================
# 3. 基础辅助函数
# ==============================================================================

def image_to_base64(fig):
    """将 matplotlib figure 转换为 base64 字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=config.FIGURE_DPI, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def _get_polygon_from_walls(walls_data: List) -> Polygon:
    if not walls_data or len(walls_data) < 3: return Polygon()
    points = [(float(wall[0]), float(wall[1])) for wall in walls_data if isinstance(wall, list) and len(wall) >= 2]
    if len(points) < 3: return Polygon()
    return Polygon(points)

def calculate_real_world_area(walls_data: List) -> Optional[float]:
    if not walls_data or len(walls_data) < 3: return None
    try:
        real_vertices = [np.array([0.0, 0.0])]
        for wall in walls_data:
            if not (isinstance(wall, list) and len(wall) == 5): continue
            x1, y1, x2, y2, real_length = [float(c) for c in wall]
            virtual_vec = np.array([x2 - x1, y2 - y1]); norm = np.linalg.norm(virtual_vec)
            if norm < 1e-6: continue
            unit_vec = virtual_vec / norm
            next_vertex = real_vertices[-1] + unit_vec * real_length
            real_vertices.append(next_vertex)
        if len(real_vertices) < 4: return None
        return Polygon(real_vertices[:-1]).area
    except: return None

def build_prompt(num_images: int, room_type: str = "room") -> str:
    image_placeholders = '\n'.join(['<image>'] * num_images)
    schema_str = '{{"real_bbox_meters": {{"width": float, "height": float}},"walls": [[...], ...],"doors": [[...], ...],"windows": [[...], ...]}}'
    
    generate_geometry_instruction = (
        "Reconstruct the room's layout based on the provided images.\n"
        "    -   **IMPORTANT RULE**: Do not oversimplify the room's shape.\n"
        "    -   **FORMAT for `walls`**: Each element must be an array: `[x1, y1, x2, y2, length_in_meters]`.\n"
        "    -   **FORMAT for `doors` and `windows`**: Each element must be an array: `[x1, y1, x2, y2, length_in_meters, parent_wall_index, length_percentage_on_wall]`."
    )
    
    return (
        f"{image_placeholders}\n\n"
        f"Analyze the images of a **{room_type}**. Generate a single structured JSON object representing the room's geometry.\n\n"
        f"**INSTRUCTION:**\n"
        f"{generate_geometry_instruction}\n\n"
        f"**OUTPUT SCHEMA:**\n"
        f"Strictly match this schema:\n"
        f"```json\n"
        f"{schema_str}\n"
        f"```\n\n"
        f"JSON:"
    )

# ==============================================================================
# 4. 3D 绘图辅助
# ==============================================================================

def _get_perpendicular_vector_3d(v):
    perp_v = np.array([-v[1], v[0]])
    norm = np.linalg.norm(perp_v)
    return perp_v / norm if norm > 1e-6 else np.array([0, 0])

def _draw_cuboid_3d(ax, vertices, color, light_source, alpha=1.0, edgecolor='k', linewidth=0.2):
    vertices = np.array(vertices)
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], 
        [vertices[4], vertices[5], vertices[6], vertices[7]], 
        [vertices[0], vertices[1], vertices[5], vertices[4]], 
        [vertices[3], vertices[2], vertices[6], vertices[7]], 
        [vertices[0], vertices[3], vertices[7], vertices[4]], 
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=to_rgba(color, alpha=alpha), 
                                         edgecolor=edgecolor, linewidth=linewidth, shade=True, lightsource=light_source))

def _draw_thick_segment_3d(ax, p1, p2, z_start, height, thickness, color, light_source):
    p1, p2 = np.array(p1), np.array(p2)
    direction_vec = p2 - p1
    if np.linalg.norm(direction_vec) < 1e-6: return
    perp_vec = _get_perpendicular_vector_3d(direction_vec) * (thickness / 2)
    v = [
        np.append(p1 - perp_vec, z_start), np.append(p1 + perp_vec, z_start), 
        np.append(p2 + perp_vec, z_start), np.append(p2 - perp_vec, z_start), 
        np.append(p1 - perp_vec, z_start + height), np.append(p1 + perp_vec, z_start + height), 
        np.append(p2 + perp_vec, z_start + height), np.append(p2 - perp_vec, z_start + height)
    ]
    _draw_cuboid_3d(ax, v, color, light_source)

def _draw_opening_panel_3d(ax, p1, p2, z, h, color):
    x1, y1 = p1; x2, y2 = p2
    verts = [(x1, y1, z), (x2, y2, z), (x2, y2, z + h), (x1, y1, z + h)]
    ax.add_collection3d(Poly3DCollection([verts], facecolors=to_rgba(color, alpha=0.65), edgecolor=None))

def _add_dimension_label_3d(ax, p1, p2, z_height, text, font_color, offset_dist=2.0, placement='center'):
    p1, p2 = np.array(p1), np.array(p2)
    mid_point = (p1 + p2) / 2
    direction_vec = p2 - p1
    if np.linalg.norm(direction_vec) < 1e-6: return
    perp_vec = _get_perpendicular_vector_3d(direction_vec)
    z_pos = config.WALL_HEIGHT if placement == 'top' else z_height
    label_pos = np.append(mid_point + perp_vec * offset_dist, z_pos)
    ax.text(*label_pos, text, color=font_color, ha='center', va='center', fontsize=config.FONT_SIZE_3D_DIM, 
            zorder=100, bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.7))

def _plot_room_on_ax_3d(ax, data, light_source, title: Optional[str] = None):
    if title:
        ax.set_title(title, fontsize=config.FONT_SIZE_3D_SUB_TITLE, pad=20)

    floor_verts = [(0, 0, 0), (config.COORDINATE_SYSTEM_MAX, 0, 0), 
                   (config.COORDINATE_SYSTEM_MAX, config.COORDINATE_SYSTEM_MAX, 0), (0, config.COORDINATE_SYSTEM_MAX, 0)]
    ax.add_collection3d(Poly3DCollection([floor_verts], facecolors=config.COLOR_FLOOR_3D, alpha=0.1))
    for i in np.arange(0, config.COORDINATE_SYSTEM_MAX + 1, config.TILE_SIZE_3D):
        ax.plot([i, i], [0, config.COORDINATE_SYSTEM_MAX], [0, 0], color=config.COLOR_FLOOR_GRID_3D, linewidth=0.5, alpha=0.5)
        ax.plot([0, config.COORDINATE_SYSTEM_MAX], [i, i], [0, 0], color=config.COLOR_FLOOR_GRID_3D, linewidth=0.5, alpha=0.5)

    walls, doors, windows = data.get('walls', []), data.get('doors', []), data.get('windows', [])
    if not walls: return

    for wall_data in walls:
        wall_p1, wall_p2, wall_len_real = np.array(wall_data[:2]), np.array(wall_data[2:4]), wall_data[4]
        openings_on_wall = []
        wall_vec, wall_vec_3d = wall_p2 - wall_p1, np.append(wall_p2 - wall_p1, 0)
        if np.linalg.norm(wall_vec) < 1e-6: continue
        wall_norm_sq = np.dot(wall_vec, wall_vec)

        for d in doors:
            op_p1 = np.array(d[:2])
            if np.linalg.norm(np.cross(wall_vec_3d, np.append(op_p1 - wall_p1, 0))) < 1 and 0 <= np.dot(op_p1 - wall_p1, wall_vec) / wall_norm_sq <= 1:
                openings_on_wall.append({'type': 'door', 'points': d[:4], 'dims': (d[4], config.DEFAULT_DOOR_H), 'z': config.DEFAULT_DOOR_Z, 'h': config.DEFAULT_DOOR_H})
        for w in windows:
            op_p1 = np.array(w[:2])
            if np.linalg.norm(np.cross(wall_vec_3d, np.append(op_p1 - wall_p1, 0))) < 1 and 0 <= np.dot(op_p1 - wall_p1, wall_vec) / wall_norm_sq <= 1:
                openings_on_wall.append({'type': 'window', 'points': w[:4], 'dims': (w[4], config.DEFAULT_WINDOW_H), 'z': config.DEFAULT_WINDOW_Z, 'h': config.DEFAULT_WINDOW_H})
        
        openings_on_wall.sort(key=lambda op: np.dot(np.array(op['points'][:2]) - wall_p1, wall_vec))
        _add_dimension_label_3d(ax, wall_p1, wall_p2, 0, f"{wall_len_real:.2f}m", config.COLOR_TEXT_WALL_3D, placement='top')
        
        current_pos = wall_p1
        for op in openings_on_wall:
            op_p1, op_p2 = np.array(op['points'][:2]), np.array(op['points'][2:4])
            _draw_thick_segment_3d(ax, current_pos, op_p1, 0, config.WALL_HEIGHT, config.WALL_THICKNESS, config.COLOR_WALL_3D, light_source)
            op_z, op_h = op['z'], op['h']
            panel_color = config.COLOR_DOOR_PANEL_3D if op['type'] == 'door' else config.COLOR_WINDOW_PANEL_3D
            _draw_opening_panel_3d(ax, op_p1, op_p2, op_z, op_h, panel_color)
            if op_z > 0: 
                _draw_thick_segment_3d(ax, op_p1, op_p2, 0, op_z, config.WALL_THICKNESS, config.COLOR_WALL_3D, light_source)
            if op_z + op_h < config.WALL_HEIGHT: 
                _draw_thick_segment_3d(ax, op_p1, op_p2, op_z + op_h, config.WALL_HEIGHT - (op_z + op_h), config.WALL_THICKNESS, config.COLOR_WALL_3D, light_source)
            op_w, _ = op['dims']
            font_color = config.COLOR_TEXT_DOOR_3D if op['type'] == 'door' else config.COLOR_TEXT_WINDOW_3D
            _add_dimension_label_3d(ax, op_p1, op_p2, op_z + op_h / 2, f"{op_w:.2f}m", font_color, config.WALL_THICKNESS * 4, 'center')
            current_pos = op_p2
        _draw_thick_segment_3d(ax, current_pos, wall_p2, 0, config.WALL_HEIGHT, config.WALL_THICKNESS, config.COLOR_WALL_3D, light_source)

# ==============================================================================
# 5. 可视化入口 (标签全部英文)
# ==============================================================================

def generate_2d_plan_base64(data: Dict) -> Optional[str]:
    try:
        plt.rcParams['font.sans-serif'] = [config.FONT_FAMILY]
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_2D, dpi=config.FIGURE_DPI)
        
        structure = data.get('structure_json', {})
        if not structure: plt.close(fig); return None

        dims = structure.get("real_bbox_meters", {})
        # [修改] 英文标题
        ax.set_title(f"Reconstruction | Size: {dims.get('width', 0):.2f}m x {dims.get('height', 0):.2f}m", fontsize=config.MAIN_TITLE_FONT_SIZE, pad=20)
        ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.6)
        room_polygon = _get_polygon_from_walls(structure.get('walls', []))

        def draw_elements(elements, color, lw, prefix, is_wall=False):
            if not elements: return
            for i, elem in enumerate(elements):
                if not (isinstance(elem, list) and len(elem) >= 5): continue
                x1, y1, x2, y2, length = [float(c) for c in elem[:5]]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=3)
                mid_x, mid_y, dx, dy = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                normal_vec = np.array([-dy, dx]); norm_len = np.linalg.norm(normal_vec)
                if norm_len > 1e-6:
                    unit_normal = normal_vec / norm_len
                    test_point = Point(mid_x + unit_normal[0] * 0.1, mid_y + unit_normal[1] * 0.1)
                    is_pointing_in = room_polygon.contains(test_point) if not room_polygon.is_empty else False
                    offset_vec = (-unit_normal if is_pointing_in else unit_normal) * config.LABEL_OFFSET if is_wall else (unit_normal if is_pointing_in else -unit_normal) * config.LABEL_OFFSET
                else: offset_vec = np.array([0, config.LABEL_OFFSET])
                text_x, text_y = mid_x + offset_vec[0], mid_y + offset_vec[1]
                angle = math.degrees(math.atan2(dy, dx))
                if angle > 90: angle -= 180
                elif angle < -90: angle += 180
                ax.text(text_x, text_y, f"{prefix}{i+1}: {length:.2f}m", fontsize=config.ANNOTATION_FONT_SIZE, color=color, ha='center', va='center', rotation=angle, rotation_mode='anchor', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=4)
                if not is_wall: ax.scatter(mid_x, mid_y, color=color, marker=config.OPENING_MARKER, s=config.OPENING_MARKER_SIZE, zorder=5)

        draw_elements(structure.get('walls', []), config.WALL_COLOR, config.LINE_WIDTH_WALL, 'W', is_wall=True)
        draw_elements(structure.get('doors', []), config.DOOR_COLOR, config.LINE_WIDTH_OPENING, 'D')
        draw_elements(structure.get('windows', []), config.WINDOW_COLOR, config.LINE_WIDTH_OPENING, 'Win')

        ax.set_xlim(-20, 120); ax.set_ylim(-20, 120)
        # [修改] 英文轴标签
        ax.set_xlabel("X-Axis", fontsize=config.AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("Y-Axis", fontsize=config.AXIS_LABEL_FONT_SIZE)
        
        # [修改] 英文图例
        handles = [
            plt.Line2D([0], [0], color=config.WALL_COLOR, linewidth=4, label='Wall'),
            plt.Line2D([0], [0], color=config.DOOR_COLOR, marker=config.OPENING_MARKER, markersize=10, linestyle='None', label='Door'),
            plt.Line2D([0], [0], color=config.WINDOW_COLOR, marker=config.OPENING_MARKER, markersize=10, linestyle='None', label='Window')
        ]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=config.LEGEND_FONT_SIZE, frameon=False)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        return image_to_base64(fig)
    except Exception as e:
        print(f"2D Drawing Error: {e}"); return None

def generate_3d_plan_base64(data: Dict, room_type: str) -> Optional[str]:
    try:
        plt.rcParams['font.sans-serif'] = [config.FONT_FAMILY]
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=config.FIGURE_SIZE_3D)
        ax = fig.add_subplot(111, projection='3d')
        # [修改] 英文标题
        main_title = f'3D Perspective View ({room_type.capitalize()})'
        fig.suptitle(main_title, fontsize=config.FONT_SIZE_3D_MAIN_TITLE, y=0.98, weight='heavy')
        light_source = LightSource(azdeg=60, altdeg=65)
        _plot_room_on_ax_3d(ax, data.get('structure_json', {}), light_source, title="AI Model Result")
        
        ax.set_xlabel('X Axis'); ax.set_ylabel('Y Axis'); ax.set_zlabel('Height')
        ax.grid(True)
        ax.set_xlim(0, config.COORDINATE_SYSTEM_MAX); ax.set_ylim(0, config.COORDINATE_SYSTEM_MAX); ax.set_zlim(0, config.WALL_HEIGHT + 2)
        ax.set_box_aspect([1, 1, 0.4])
        ax.view_init(elev=30, azim=-120)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return image_to_base64(fig)
    except Exception as e:
        print(f"3D Drawing Error: {e}"); return None

# ==============================================================================
# 6. FastAPI 核心逻辑 (加入统计功能)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_engine
    try:
        model_engine = PtEngine(MODEL_PATH)
        print("✅ Model engine ready.")
    except Exception as e:
        print(f"❌ Model load error: {e}"); raise e
    yield
    torch.cuda.empty_cache()

app = FastAPI(title="Room Layout API (2D + 3D)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    room_type: str = "bedroom"
    images: List[str]
    return_2d: bool = True
    return_3d: bool = True

async def _run_inference_core(pil_images: List[Image.Image], room_type: str, return_2d: bool, return_3d: bool):
    global model_engine
    if not model_engine: raise HTTPException(status_code=500, detail="Model uninitialized")
    
    resized_images = [img.resize((336, 336), Image.Resampling.LANCZOS) for img in pil_images]
    prompt = build_prompt(len(resized_images), room_type)
    
    t_start = time.time()
    async with inference_lock:
        try:
            req = InferRequest(messages=[{'role': 'user', 'content': prompt}], images=resized_images)
            resp = model_engine.infer([req], RequestConfig(temperature=0, stream=False))
            resp_str = resp[0].choices[0].message.content
        except Exception as e: raise HTTPException(500, detail=f"Inference error: {e}")
    t_infer = time.time() - t_start

    try:
        match = re.search(r'```json\s*(\{.*?\})\s*```', resp_str, re.DOTALL)
        clean_json = match.group(1) if match else resp_str[resp_str.find('{'):resp_str.rfind('}')+1]
        parsed = json.loads(clean_json)
        if 'structure_json' not in parsed: parsed = {'structure_json': parsed}
    except:
        return {"status": "error", "raw": resp_str}

    # 1. 计算物理指标
    structure = parsed.get('structure_json', {})
    walls = structure.get('walls', [])
    doors = structure.get('doors', [])
    windows = structure.get('windows', [])
    
    area = calculate_real_world_area(walls)

    # 2. [新增] 构造统计报告 statistics
    wall_lengths = [round(w[4], 2) for w in walls if len(w) >= 5]
    door_lengths = [round(d[4], 2) for d in doors if len(d) >= 5]
    window_lengths = [round(w[4], 2) for w in windows if len(w) >= 5]
    
    statistics = {
        "counts": {
            "walls": len(walls),
            "doors": len(doors),
            "windows": len(windows)
        },
        "lengths": {
            "wall_details": wall_lengths,
            "door_details": door_lengths,
            "window_details": window_lengths,
            "total_wall_length": round(sum(wall_lengths), 2)
        },
        "area_sqm": round(area, 2) if area else 0
    }
    
    res = {
        "status": "success", 
        "room_type": room_type, 
        "inference_time": round(t_infer, 1), 
        "statistics": statistics, # 数值结果统计
        "area": area, 
        "data": parsed
    }

    if return_2d: res["visualization_2d"] = generate_2d_plan_base64(parsed)
    if return_3d: res["visualization_3d"] = generate_3d_plan_base64(parsed, room_type)

    return res

# --- 路由 ---

@app.post("/predict")
async def predict_layout(
    files: List[UploadFile] = File(...), 
    room_type: str = "room",
    return_2d: bool = True,
    return_3d: bool = False 
):
    pil_images = []
    try:
        for file in files:
            c = await file.read()
            pil_images.append(Image.open(io.BytesIO(c)).convert('RGB'))
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))
    return await _run_inference_core(pil_images, room_type, return_2d, return_3d)

@app.post("/predict_base64")
async def predict_layout_base64(data: ImageRequest):
    pil_images = []
    try:
        for img_str in data.images:
            if "," in img_str: img_str = img_str.split(",")[1]
            image_bytes = base64.b64decode(img_str)
            pil_images.append(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    except Exception as e: raise HTTPException(status_code=400, detail=f"Invalid base64")
    return await _run_inference_core(pil_images, data.room_type, data.return_2d, data.return_3d)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


# import os
# import json
# import io
# import base64
# import time
# import math
# import re
# import asyncio
# from typing import List, Dict, Optional
# from contextlib import asynccontextmanager

# import torch
# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware  # [NEW] 引入 CORS

# # Matplotlib 设置 (无头模式)
# import matplotlib
# matplotlib.use('Agg') 
# import matplotlib.pyplot as plt
# # 3D 绘图相关引用
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from matplotlib.colors import LightSource, to_rgba
# from shapely.geometry import Polygon, Point

# # Swift 模型相关引用
# from swift.llm import PtEngine, InferRequest, RequestConfig

# # ==============================================================================
# # 1. 全局配置与常量 (Config)
# # ==============================================================================
# MODEL_PATH = '/dataz/JM/checkpoint_tem/v0.0.8_unorder/v0-20250904-193049/Qwen2.5-VL-7B-Instruct'

# class Config:
#     # --- 字体与文本 ---
#     FONT_FAMILY = 'WenQuanYi Zen Hei' 
#     MAIN_TITLE_FONT_SIZE, SUB_TITLE_FONT_SIZE = 28, 24
#     AXIS_LABEL_FONT_SIZE, LEGEND_FONT_SIZE, ANNOTATION_FONT_SIZE = 20, 22, 18
    
#     # --- 2D 绘图参数 ---
#     WALL_COLOR, DOOR_COLOR, WINDOW_COLOR = 'black', 'royalblue', 'skyblue'
#     OPENING_MARKER, OPENING_MARKER_SIZE = 'o', 80
#     LINE_WIDTH_WALL, LINE_WIDTH_OPENING = 4.0, 8.0
#     FIGURE_SIZE_2D, FIGURE_DPI = (16, 16), 120
#     LABEL_OFFSET = 3.5

#     # --- 3D 绘图参数 ---
#     FIGURE_SIZE_3D = (12, 12)
#     FONT_SIZE_3D_MAIN_TITLE, FONT_SIZE_3D_SUB_TITLE, FONT_SIZE_3D_DIM = 28, 22, 14
    
#     # 物理尺寸
#     WALL_HEIGHT = 2.7
#     WALL_THICKNESS = 0.25
#     COORDINATE_SYSTEM_MAX = 100.0
#     TILE_SIZE_3D = 10
    
#     # 门窗高度
#     DEFAULT_DOOR_Z, DEFAULT_DOOR_H = 0.0, 2.1
#     DEFAULT_WINDOW_Z, DEFAULT_WINDOW_H = 0.9, 1.5
    
#     # 3D 颜色
#     COLOR_WALL_3D = '#EAE0D5'
#     COLOR_DOOR_PANEL_3D = '#8B4513'
#     COLOR_WINDOW_PANEL_3D = '#ADD8E6'
#     COLOR_FLOOR_3D = '#F5F5F5'
#     COLOR_FLOOR_GRID_3D = '#D3D3D3'
#     COLOR_TEXT_WALL_3D = '#5D4037'
#     COLOR_TEXT_DOOR_3D = '#BF360C'
#     COLOR_TEXT_WINDOW_3D = '#0D47A1'

# config = Config()

# # ==============================================================================
# # 2. 全局变量
# # ==============================================================================
# model_engine = None
# inference_lock = asyncio.Lock()

# # ==============================================================================
# # 3. 基础辅助函数
# # ==============================================================================

# def image_to_base64(fig):
#     """将 matplotlib figure 转换为 base64 字符串"""
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', dpi=config.FIGURE_DPI, bbox_inches='tight')
#     buf.seek(0)
#     img_str = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close(fig)
#     return img_str

# def _get_polygon_from_walls(walls_data: List) -> Polygon:
#     if not walls_data or len(walls_data) < 3: return Polygon()
#     points = [(float(wall[0]), float(wall[1])) for wall in walls_data if isinstance(wall, list) and len(wall) >= 2]
#     if len(points) < 3: return Polygon()
#     return Polygon(points)

# def calculate_real_world_area(walls_data: List) -> Optional[float]:
#     if not walls_data or len(walls_data) < 3: return None
#     try:
#         real_vertices = [np.array([0.0, 0.0])]
#         for wall in walls_data:
#             if not (isinstance(wall, list) and len(wall) == 5): continue
#             x1, y1, x2, y2, real_length = [float(c) for c in wall]
#             virtual_vec = np.array([x2 - x1, y2 - y1]); norm = np.linalg.norm(virtual_vec)
#             if norm < 1e-6: continue
#             unit_vec = virtual_vec / norm
#             next_vertex = real_vertices[-1] + unit_vec * real_length
#             real_vertices.append(next_vertex)
#         if len(real_vertices) < 4: return None
#         return Polygon(real_vertices[:-1]).area
#     except: return None

# def build_prompt(num_images: int, room_type: str = "room") -> str:
#     room_type_map = {'卧室': 'bedroom', '客厅': 'living room', '厨房': 'kitchen', '卫生间': 'bathroom'}
#     room_type_en = next((v for k, v in room_type_map.items() if k in room_type), room_type)
#     image_placeholders = '\n'.join(['<image>'] * num_images)
#     schema_str = '{{"real_bbox_meters": {{"width": float, "height": float}},"walls": [[...], ...],"doors": [[...], ...],"windows": [[...], ...]}}'
    
#     generate_geometry_instruction = (
#         "Reconstruct the room's layout based on the provided images.\n"
#         "    -   **IMPORTANT RULE**: Do not oversimplify the room's shape.\n"
#         "    -   **FORMAT for `walls`**: Each element must be an array: `[x1, y1, x2, y2, length_in_meters]`.\n"
#         "    -   **FORMAT for `doors` and `windows`**: Each element must be an array: `[x1, y1, x2, y2, length_in_meters, parent_wall_index, length_percentage_on_wall]`."
#     )
    
#     return (
#         f"{image_placeholders}\n\n"
#         f"Analyze the provided unordered images of a **{room_type_en}**. Your task is to generate a single structured JSON object representing the room's complete geometry in a 100x100 coordinate system.\n\n"
#         f"**INSTRUCTION:**\n"
#         f"{generate_geometry_instruction}\n\n"
#         f"**OUTPUT SCHEMA:**\n"
#         f"Your output must strictly be a single JSON object matching this schema:\n"
#         f"```json\n"
#         f"{schema_str}\n"
#         f"```\n\n"
#         f"JSON:"
#     )

# # ==============================================================================
# # 4. [核心] 3D 绘图辅助函数
# # ==============================================================================

# def _get_perpendicular_vector_3d(v):
#     perp_v = np.array([-v[1], v[0]])
#     norm = np.linalg.norm(perp_v)
#     return perp_v / norm if norm > 1e-6 else np.array([0, 0])

# def _draw_cuboid_3d(ax, vertices, color, light_source, alpha=1.0, edgecolor='k', linewidth=0.2):
#     vertices = np.array(vertices)
#     faces = [
#         [vertices[0], vertices[1], vertices[2], vertices[3]], 
#         [vertices[4], vertices[5], vertices[6], vertices[7]], 
#         [vertices[0], vertices[1], vertices[5], vertices[4]], 
#         [vertices[3], vertices[2], vertices[6], vertices[7]], 
#         [vertices[0], vertices[3], vertices[7], vertices[4]], 
#         [vertices[1], vertices[2], vertices[6], vertices[5]]
#     ]
#     ax.add_collection3d(Poly3DCollection(faces, facecolors=to_rgba(color, alpha=alpha), 
#                                          edgecolor=edgecolor, linewidth=linewidth, shade=True, lightsource=light_source))

# def _draw_thick_segment_3d(ax, p1, p2, z_start, height, thickness, color, light_source):
#     p1, p2 = np.array(p1), np.array(p2)
#     direction_vec = p2 - p1
#     if np.linalg.norm(direction_vec) < 1e-6: return
#     perp_vec = _get_perpendicular_vector_3d(direction_vec) * (thickness / 2)
#     v = [
#         np.append(p1 - perp_vec, z_start), np.append(p1 + perp_vec, z_start), 
#         np.append(p2 + perp_vec, z_start), np.append(p2 - perp_vec, z_start), 
#         np.append(p1 - perp_vec, z_start + height), np.append(p1 + perp_vec, z_start + height), 
#         np.append(p2 + perp_vec, z_start + height), np.append(p2 - perp_vec, z_start + height)
#     ]
#     _draw_cuboid_3d(ax, v, color, light_source)

# def _draw_opening_panel_3d(ax, p1, p2, z, h, color):
#     x1, y1 = p1; x2, y2 = p2
#     verts = [(x1, y1, z), (x2, y2, z), (x2, y2, z + h), (x1, y1, z + h)]
#     ax.add_collection3d(Poly3DCollection([verts], facecolors=to_rgba(color, alpha=0.65), edgecolor=None))

# def _add_dimension_label_3d(ax, p1, p2, z_height, text, font_color, offset_dist=2.0, placement='center'):
#     p1, p2 = np.array(p1), np.array(p2)
#     mid_point = (p1 + p2) / 2
#     direction_vec = p2 - p1
#     if np.linalg.norm(direction_vec) < 1e-6: return
#     perp_vec = _get_perpendicular_vector_3d(direction_vec)
#     z_pos = config.WALL_HEIGHT if placement == 'top' else z_height
#     label_pos = np.append(mid_point + perp_vec * offset_dist, z_pos)
#     ax.text(*label_pos, text, color=font_color, ha='center', va='center', fontsize=config.FONT_SIZE_3D_DIM, 
#             zorder=100, bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.7))

# def _plot_room_on_ax_3d(ax, data, light_source, title: Optional[str] = None):
#     if title:
#         ax.set_title(title, fontsize=config.FONT_SIZE_3D_SUB_TITLE, pad=20)

#     # 绘制地板和网格
#     floor_verts = [(0, 0, 0), (config.COORDINATE_SYSTEM_MAX, 0, 0), 
#                    (config.COORDINATE_SYSTEM_MAX, config.COORDINATE_SYSTEM_MAX, 0), (0, config.COORDINATE_SYSTEM_MAX, 0)]
#     ax.add_collection3d(Poly3DCollection([floor_verts], facecolors=config.COLOR_FLOOR_3D, alpha=0.1))
#     for i in np.arange(0, config.COORDINATE_SYSTEM_MAX + 1, config.TILE_SIZE_3D):
#         ax.plot([i, i], [0, config.COORDINATE_SYSTEM_MAX], [0, 0], color=config.COLOR_FLOOR_GRID_3D, linewidth=0.5, alpha=0.5)
#         ax.plot([0, config.COORDINATE_SYSTEM_MAX], [i, i], [0, 0], color=config.COLOR_FLOOR_GRID_3D, linewidth=0.5, alpha=0.5)

#     walls, doors, windows = data.get('walls', []), data.get('doors', []), data.get('windows', [])
#     if not walls: return

#     for wall_data in walls:
#         wall_p1, wall_p2, wall_len_real = np.array(wall_data[:2]), np.array(wall_data[2:4]), wall_data[4]
        
#         openings_on_wall = []
#         wall_vec, wall_vec_3d = wall_p2 - wall_p1, np.append(wall_p2 - wall_p1, 0)
#         if np.linalg.norm(wall_vec) < 1e-6: continue
#         wall_norm_sq = np.dot(wall_vec, wall_vec)

#         for d in doors:
#             op_p1 = np.array(d[:2])
#             if np.linalg.norm(np.cross(wall_vec_3d, np.append(op_p1 - wall_p1, 0))) < 1 and 0 <= np.dot(op_p1 - wall_p1, wall_vec) / wall_norm_sq <= 1:
#                 openings_on_wall.append({'type': 'door', 'points': d[:4], 'dims': (d[4], config.DEFAULT_DOOR_H), 'z': config.DEFAULT_DOOR_Z, 'h': config.DEFAULT_DOOR_H})
#         for w in windows:
#             op_p1 = np.array(w[:2])
#             if np.linalg.norm(np.cross(wall_vec_3d, np.append(op_p1 - wall_p1, 0))) < 1 and 0 <= np.dot(op_p1 - wall_p1, wall_vec) / wall_norm_sq <= 1:
#                 openings_on_wall.append({'type': 'window', 'points': w[:4], 'dims': (w[4], config.DEFAULT_WINDOW_H), 'z': config.DEFAULT_WINDOW_Z, 'h': config.DEFAULT_WINDOW_H})
        
#         openings_on_wall.sort(key=lambda op: np.dot(np.array(op['points'][:2]) - wall_p1, wall_vec))
        
#         _add_dimension_label_3d(ax, wall_p1, wall_p2, 0, f"{wall_len_real:.2f}m", config.COLOR_TEXT_WALL_3D, placement='top')
        
#         current_pos = wall_p1
#         for op in openings_on_wall:
#             op_p1, op_p2 = np.array(op['points'][:2]), np.array(op['points'][2:4])
#             _draw_thick_segment_3d(ax, current_pos, op_p1, 0, config.WALL_HEIGHT, config.WALL_THICKNESS, config.COLOR_WALL_3D, light_source)
            
#             op_z, op_h = op['z'], op['h']
#             panel_color = config.COLOR_DOOR_PANEL_3D if op['type'] == 'door' else config.COLOR_WINDOW_PANEL_3D
#             _draw_opening_panel_3d(ax, op_p1, op_p2, op_z, op_h, panel_color)
            
#             if op_z > 0: 
#                 _draw_thick_segment_3d(ax, op_p1, op_p2, 0, op_z, config.WALL_THICKNESS, config.COLOR_WALL_3D, light_source)
#             if op_z + op_h < config.WALL_HEIGHT: 
#                 _draw_thick_segment_3d(ax, op_p1, op_p2, op_z + op_h, config.WALL_HEIGHT - (op_z + op_h), config.WALL_THICKNESS, config.COLOR_WALL_3D, light_source)
            
#             op_w, _ = op['dims']
#             font_color = config.COLOR_TEXT_DOOR_3D if op['type'] == 'door' else config.COLOR_TEXT_WINDOW_3D
#             _add_dimension_label_3d(ax, op_p1, op_p2, op_z + op_h / 2, f"{op_w:.2f}m", font_color, config.WALL_THICKNESS * 4, 'center')
#             current_pos = op_p2
        
#         _draw_thick_segment_3d(ax, current_pos, wall_p2, 0, config.WALL_HEIGHT, config.WALL_THICKNESS, config.COLOR_WALL_3D, light_source)

# # ==============================================================================
# # 5. 绘图入口函数 (2D 和 3D)
# # ==============================================================================

# def generate_2d_plan_base64(data: Dict) -> Optional[str]:
#     """生成 2D 平面图"""
#     try:
#         plt.rcParams['font.sans-serif'] = [config.FONT_FAMILY]
#         plt.rcParams['axes.unicode_minus'] = False
#         fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_2D, dpi=config.FIGURE_DPI)
        
#         structure = data.get('structure_json', {})
#         if not structure: plt.close(fig); return None

#         dims = structure.get("real_bbox_meters", {})
#         ax.set_title(f"模型预测 | 尺寸: {dims.get('width', 0):.2f}m x {dims.get('height', 0):.2f}m", fontsize=config.MAIN_TITLE_FONT_SIZE, pad=20)
#         ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle='--', alpha=0.6)
#         room_polygon = _get_polygon_from_walls(structure.get('walls', []))

#         def draw_elements(elements, color, lw, prefix, is_wall=False):
#             if not elements: return
#             for i, elem in enumerate(elements):
#                 if not (isinstance(elem, list) and len(elem) >= 5): continue
#                 x1, y1, x2, y2, length = [float(c) for c in elem[:5]]
#                 ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=3)
#                 mid_x, mid_y, dx, dy = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
#                 normal_vec = np.array([-dy, dx]); norm_len = np.linalg.norm(normal_vec)
#                 if norm_len > 1e-6:
#                     unit_normal = normal_vec / norm_len
#                     test_point = Point(mid_x + unit_normal[0] * 0.1, mid_y + unit_normal[1] * 0.1)
#                     is_pointing_in = room_polygon.contains(test_point) if not room_polygon.is_empty else False
#                     offset_vec = (-unit_normal if is_pointing_in else unit_normal) * config.LABEL_OFFSET if is_wall else (unit_normal if is_pointing_in else -unit_normal) * config.LABEL_OFFSET
#                 else: offset_vec = np.array([0, config.LABEL_OFFSET])
#                 text_x, text_y = mid_x + offset_vec[0], mid_y + offset_vec[1]
#                 angle = math.degrees(math.atan2(dy, dx))
#                 if angle > 90: angle -= 180
#                 elif angle < -90: angle += 180
#                 ax.text(text_x, text_y, f"{prefix}{i+1}: {length:.2f}m", fontsize=config.ANNOTATION_FONT_SIZE, color=color, ha='center', va='center', rotation=angle, rotation_mode='anchor', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=4)
#                 if not is_wall: ax.scatter(mid_x, mid_y, color=color, marker=config.OPENING_MARKER, s=config.OPENING_MARKER_SIZE, zorder=5)

#         draw_elements(structure.get('walls', []), config.WALL_COLOR, config.LINE_WIDTH_WALL, 'W', is_wall=True)
#         draw_elements(structure.get('doors', []), config.DOOR_COLOR, config.LINE_WIDTH_OPENING, 'D')
#         draw_elements(structure.get('windows', []), config.WINDOW_COLOR, config.LINE_WIDTH_OPENING, 'Win')

#         ax.set_xlim(-20, 120); ax.set_ylim(-20, 120)
#         ax.set_xlabel("X 坐标", fontsize=config.AXIS_LABEL_FONT_SIZE); ax.set_ylabel("Y 坐标", fontsize=config.AXIS_LABEL_FONT_SIZE)
        
#         handles = [
#             plt.Line2D([0], [0], color=config.WALL_COLOR, linewidth=4, label='墙体'),
#             plt.Line2D([0], [0], color=config.DOOR_COLOR, marker=config.OPENING_MARKER, markersize=10, linestyle='None', label='门'),
#             plt.Line2D([0], [0], color=config.WINDOW_COLOR, marker=config.OPENING_MARKER, markersize=10, linestyle='None', label='窗户')
#         ]
#         fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=config.LEGEND_FONT_SIZE, frameon=False)
#         plt.tight_layout(rect=[0, 0.08, 1, 0.95])
#         return image_to_base64(fig)
#     except Exception as e:
#         print(f"2D Drawing Error: {e}"); return None

# def generate_3d_plan_base64(data: Dict, room_type: str) -> Optional[str]:
#     """生成 3D 户型图"""
#     try:
#         plt.rcParams['font.sans-serif'] = [config.FONT_FAMILY]
#         plt.rcParams['axes.unicode_minus'] = False
        
#         fig = plt.figure(figsize=config.FIGURE_SIZE_3D)
#         ax = fig.add_subplot(111, projection='3d')
        
#         main_title = f'模型预测 3D 视图 ({room_type})'
#         fig.suptitle(main_title, fontsize=config.FONT_SIZE_3D_MAIN_TITLE, y=0.98, weight='heavy')
        
#         light_source = LightSource(azdeg=60, altdeg=65)
#         _plot_room_on_ax_3d(ax, data.get('structure_json', {}), light_source, title="AI Prediction")
        
#         fixed_angle_elev, fixed_angle_azim = 30, -120
#         ax.set_xlabel('X Axis'); ax.set_ylabel('Y Axis'); ax.set_zlabel('Height')
#         ax.grid(True)
#         ax.set_xlim(0, config.COORDINATE_SYSTEM_MAX)
#         ax.set_ylim(0, config.COORDINATE_SYSTEM_MAX)
#         ax.set_zlim(0, config.WALL_HEIGHT + 2)
#         ax.set_box_aspect([1, 1, 0.4])
#         ax.view_init(elev=fixed_angle_elev, azim=fixed_angle_azim)
        
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         return image_to_base64(fig)
#     except Exception as e:
#         print(f"3D Drawing Error: {e}"); import traceback; traceback.print_exc(); return None

# # ==============================================================================
# # 6. FastAPI 生命周期与路由
# # ==============================================================================

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global model_engine
#     print("\n" + "="*30)
#     print(f"正在加载模型: {MODEL_PATH}")
#     try:
#         model_engine = PtEngine(MODEL_PATH)
#         print("✅ 模型加载成功！服务准备就绪。")
#     except Exception as e:
#         print(f"❌ 模型加载失败: {e}")
#         raise e
#     yield
#     print("服务正在关闭..."); torch.cuda.empty_cache()

# app = FastAPI(title="Room Layout API (2D + 3D)", lifespan=lifespan)

# # [NEW] 配置 CORS - 允许所有来源，这是 App 能访问的关键
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # [NEW] 定义 JSON 请求体结构
# class ImageRequest(BaseModel):
#     room_type: str = "bedroom"
#     images: List[str] # Base64 字符串列表
#     return_2d: bool = True
#     return_3d: bool = True

# # [NEW] 核心推理逻辑解耦，供文件接口和 Base64 接口共用
# async def _run_inference_core(pil_images: List[Image.Image], room_type: str, return_2d: bool, return_3d: bool):
#     global model_engine
#     if not model_engine: raise HTTPException(status_code=500, detail="Model uninitialized")
    
#     # 统一调整图片大小
#     resized_images = [img.resize((336, 336), Image.Resampling.LANCZOS) for img in pil_images]
    
#     prompt = build_prompt(len(resized_images), room_type)
    
#     t_start = time.time()
#     async with inference_lock:
#         try:
#             req = InferRequest(messages=[{'role': 'user', 'content': prompt}], images=resized_images)
#             resp = model_engine.infer([req], RequestConfig(temperature=0, stream=False))
#             resp_str = resp[0].choices[0].message.content
#         except Exception as e: raise HTTPException(500, detail=f"Inference error: {e}")
#     t_infer = time.time() - t_start

#     try:
#         match = re.search(r'```json\s*(\{.*?\})\s*```', resp_str, re.DOTALL)
#         clean_json = match.group(1) if match else resp_str[resp_str.find('{'):resp_str.rfind('}')+1]
#         parsed = json.loads(clean_json)
#         if 'structure_json' not in parsed: parsed = {'structure_json': parsed}
#     except:
#         return {"status": "error", "raw": resp_str}

#     area = calculate_real_world_area(parsed['structure_json'].get('walls', []))
    
#     res = {
#         "status": "success", 
#         "room_type": room_type, 
#         "inference_time": t_infer, 
#         "area": area, 
#         "data": parsed
#     }

#     if return_2d: res["visualization_2d"] = generate_2d_plan_base64(parsed)
#     if return_3d: res["visualization_3d"] = generate_3d_plan_base64(parsed, room_type)

#     return res

# # --- 路由定义 ---

# @app.post("/predict")
# async def predict_layout(
#     files: List[UploadFile] = File(...), 
#     room_type: str = "room",
#     return_2d: bool = True,
#     return_3d: bool = False 
# ):
#     """旧接口：支持 Form-Data 文件上传"""
#     pil_images = []
#     try:
#         for file in files:
#             c = await file.read()
#             pil_images.append(Image.open(io.BytesIO(c)).convert('RGB'))
#     except Exception as e: raise HTTPException(status_code=400, detail=str(e))
    
#     if not pil_images: raise HTTPException(400, "No images")
#     return await _run_inference_core(pil_images, room_type, return_2d, return_3d)

# @app.post("/predict_base64")
# async def predict_layout_base64(data: ImageRequest):
#     """[NEW] 新接口：支持 JSON Base64 列表上传 (推荐给 App 使用)"""
#     pil_images = []
#     try:
#         for img_str in data.images:
#             # 去除 data:image/jpeg;base64, 前缀（如果存在）
#             if "," in img_str:
#                 img_str = img_str.split(",")[1]
#             image_bytes = base64.b64decode(img_str)
#             pil_images.append(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
#     if not pil_images: raise HTTPException(400, "No images provided")
#     return await _run_inference_core(pil_images, data.room_type, data.return_2d, data.return_3d)

# if __name__ == "__main__":
#     import uvicorn
#     # 允许局域网访问，方便真机调试
#     uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)