"""整图缩略预览；rasterio 使用 out_shape，不按源图尺寸分配 RGB。"""
from pathlib import Path
from PIL import Image, ImageDraw
from ..io import open_image, resolve_path


def read_preview(image: dict, config) -> tuple[Image.Image, float]:
    """按 overview_max_side 读取整图缩略 RGB，返回 (图像, 缩放比例)。"""
    scale = min(1., config.overview_max_side/max(image['width'], image['height']))
    size = (max(1, round(image['width']*scale)), max(1, round(image['height']*scale)))
    with open_image(image['path'], config.pillow_max_image_pixels) as source:
        if source.raster is not None:
            from rasterio.enums import Resampling
            pixels = source.raster.read([1, 2, 3], out_shape=(3, size[1], size[0]), resampling=Resampling.bilinear)
            preview = Image.fromarray(pixels.transpose(1, 2, 0))
        else:
            preview = source.pillow.convert('RGB')
            preview.thumbnail(size)
    return preview, scale


def render_global(image: dict, output: str | Path, config, *, edges=(), dropped=(), blocks=(), tiles=()) -> str:
    preview, scale = read_preview(image, config)
    draw = ImageDraw.Draw(preview)
    for tile in tiles:
        draw.rectangle([tile[k]*scale for k in ('x0', 'y0', 'x1', 'y1')], outline=(160, 160, 255))
    for index, edge in enumerate(list(dropped)+list(edges)):
        color = (255, 60, 60) if index < len(dropped) else ((30+index*73 % 200), (120+index*41 % 130), (80+index*97 % 170))
        points = [(x*scale, y*scale) for x, y in edge['polyline_xy']]
        draw.line(points, fill=color, width=config.annotation_line_width)
        middle = points[len(points)//2]
        draw.text((middle[0]+config.annotation_offset_px, middle[1]+config.annotation_offset_px),
                  f"E{edge['edge_id']}", fill=color)
    for block in blocks:
        draw.rectangle([v*scale for v in block['global_bbox']], outline=(255, 220, 0))
        x, y = block['global_xy']
        draw.text((x*scale, y*scale), block['block_id'], fill=(255, 220, 0))
    path = resolve_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path)
    return str(path)


def render_block_layout(image: dict, output: str | Path, config, *, edges=(), blocks=(), title: str = '') -> str:
    """布块对照图：道路白线；同一条边的块用同一种半透明颜色和粗边框，标块 ID；顶部标题。"""
    from PIL import ImageFont
    preview, scale = read_preview(image, config)
    side = max(preview.size)
    width = max(2, side//900)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', max(11, side//300))
        head = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', max(14, side//120))
    except OSError:
        font = head = ImageFont.load_default()
    base = preview.convert('RGBA')
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for edge in edges:
        draw.line([(x*scale, y*scale) for x, y in edge['polyline_xy']], fill=(255, 255, 255, 220), width=width)
    palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
               (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]
    order = {edge_id: i for i, edge_id in enumerate(sorted({b['edge_id'] for b in blocks}, key=str))}
    colors = [palette[order[b['edge_id']] % len(palette)] for b in blocks]
    # 先画全部填色、再画全部边框：重叠处后画的块不会盖住先画块的边框，每个方框完整可见。
    for block, color in zip(blocks, colors):
        draw.rectangle([v*scale for v in block['global_bbox']], fill=color+(40,))
    result_fill = Image.alpha_composite(base, layer)
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for block, color in zip(blocks, colors):
        draw.rectangle([v*scale for v in block['global_bbox']], outline=color+(255,), width=width)
    for block in blocks:
        x, y = block['global_xy']
        draw.text((x*scale, y*scale), block['block_id'], fill=(255, 255, 255, 255), font=font, anchor='mm',
                  stroke_width=2, stroke_fill=(0, 0, 0, 255))
    result = Image.alpha_composite(result_fill, layer).convert('RGB')
    band = head.size+16
    canvas = Image.new('RGB', (result.width, result.height+band), 'white')
    canvas.paste(result, (0, band))
    ImageDraw.Draw(canvas).text((10, 8), title, fill='black', font=head)
    path = resolve_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def render_road_topology(image: dict, output: str | Path, config, *, edges=(), nodes=(), title: str = '') -> str:
    """最终道路拓扑：粗线（每条边一种颜色、黑色描边）、路口/端点圆点、边 ID；顶部标题。"""
    from PIL import ImageFont
    preview, scale = read_preview(image, config)
    side = max(preview.size)
    width = max(4, side//180)
    bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    try:
        font, head = ImageFont.truetype(bold, max(12, side//200)), ImageFont.truetype(bold, max(14, side//120))
    except OSError:
        font = head = ImageFont.load_default()
    # 底图压暗，突出道路
    base = Image.blend(preview.convert('RGB'), Image.new('RGB', preview.size, 'black'), .35)
    draw = ImageDraw.Draw(base)
    palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
               (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]
    ordered = sorted(edges, key=lambda e: int(e['edge_id']) if str(e['edge_id']).isdigit() else str(e['edge_id']))
    lines = [[(x*scale, y*scale) for x, y in e['polyline_xy']] for e in ordered]
    # 描边区分来源：SAM-Road 黑色描边；语义补全（semantic/mixed）白色描边
    for edge, points in zip(ordered, lines):
        casing = (255, 255, 255) if edge.get('source') in ('semantic', 'mixed') else (0, 0, 0)
        draw.line(points, fill=casing, width=width+4, joint='curve')
    for index, points in enumerate(lines):
        draw.line(points, fill=palette[index % len(palette)], width=width, joint='curve')
    degree: dict[str, int] = {}
    for edge in edges:
        for key in ('u', 'v'):
            degree[edge[key]] = degree.get(edge[key], 0)+1
    for node in nodes:
        x, y = node['xy'][0]*scale, node['xy'][1]*scale
        junction = degree.get(node['node_id'], 0) >= 3
        r = width*1.3 if junction else width*.8
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255) if junction else (40, 40, 40),
                     outline=(0, 0, 0) if junction else (255, 255, 255), width=max(1, width//3))
    for edge, points in zip(ordered, lines):
        x, y = points[len(points)//2] if len(points) > 2 else ((points[0][0]+points[-1][0])/2, (points[0][1]+points[-1][1])/2)
        draw.text((x, y), f"E{edge['edge_id']}", fill=(255, 255, 255), font=font, anchor='mm',
                  stroke_width=3, stroke_fill=(0, 0, 0))
    band = head.size+16
    canvas = Image.new('RGB', (base.width, base.height+band), 'white')
    canvas.paste(base, (0, band))
    legend = ImageDraw.Draw(canvas)
    legend.text((10, 8), title, fill='black', font=head)
    x = 20+int(legend.textlength(title, font=head))+30
    r = head.size//2
    for label, fill, outline in (('junction (degree >= 3)', (255, 255, 255), (0, 0, 0)), ('end / bend node', (40, 40, 40), (0, 0, 0))):
        legend.ellipse([x, 8+r//2, x+2*r, 8+r//2+2*r], fill=fill, outline=outline, width=2)
        legend.text((x+2*r+8, 8), label, fill='black', font=head)
        x += 2*r+8+int(legend.textlength(label, font=head))+30
    if any(e.get('source') in ('semantic', 'mixed') for e in edges):
        for label, casing in (('SAM-Road (black casing)', (0, 0, 0)), ('semantic-added (white casing)', (160, 160, 160))):
            legend.line([x, 8+head.size//2, x+40, 8+head.size//2], fill=casing, width=8)
            legend.line([x, 8+head.size//2, x+40, 8+head.size//2], fill=(230, 25, 75), width=4)
            legend.text((x+48, 8), label, fill='black', font=head)
            x += 48+int(legend.textlength(label, font=head))+30
    path = resolve_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)
