"""整幅 GeoTIFF 直接生成相对深度 GeoTIFF：CRS、仿射变换和像素网格与输入完全一致。

一次前向覆盖全图，细节上限由 --input-size 决定。输出是无单位相对逆深度（越大越近），
不是 DSM 或高程。
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

from agent.paths import resolve_path
from agent.perception_config import load_perception_config
from percection_tool.factory import create_depth_estimator

# 3780 是 RTX 3090（24G）FP32 上验证过的最高档，峰值约 18.7 GiB；显存不足时调小 --input-size。
DEFAULT_INPUT_SIZE = 3780
MAX_PROCESSED_PIXELS = 16_000_000


def run(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image',default='input_geo_img/kitti_0930_0033.tif')
    parser.add_argument('--output',help='默认 output/perception_tools/whole_image/<图片名>_depth.tif')
    parser.add_argument('--input-size',type=int,default=DEFAULT_INPUT_SIZE,help='短边处理尺寸，须为 14 的倍数')
    args = parser.parse_args(argv)

    import rasterio
    from agent.map_preprocess.io import open_image
    settings = load_perception_config('agent/perception_tools.yaml').depth
    with open_image(args.image,settings.max_input_pixels) as source:
        if source.raster is None or source.raster.crs is None:
            raise ValueError('输入没有地理参考，无法保持经纬度')
        crs,transform = source.raster.crs,source.raster.transform
        bbox = [0,0,source.width,source.height]
        rgb,valid = source.read(bbox),source.valid_mask(bbox)
    height,width = rgb.shape[:2]

    tool = create_depth_estimator(replace(settings,input_size=args.input_size,
        max_input_pixels=max(settings.max_input_pixels,width*height),max_processed_pixels=MAX_PROCESSED_PIXELS))
    try:
        depth = tool.predict(rgb,valid_mask=valid)['depth']
    finally:
        tool.close()

    output = resolve_path(args.output or f'output/perception_tools/whole_image/{Path(args.image).stem}_depth.tif')
    output.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(output,'w',driver='GTiff',width=width,height=height,count=1,dtype='float32',crs=crs,
                       transform=transform,nodata=float('nan'),tiled=True,blockxsize=256,blockysize=256,
                       compress='deflate',predictor=3,BIGTIFF='IF_SAFER') as target:
        target.write(depth,1)
        target.update_tags(representation='relative_inverse_depth',larger_is='nearer',is_metric='false',
                           is_elevation='false',input_size=str(args.input_size))
    return output


def main(argv: list[str] | None = None) -> int:
    # 大 input_size 下碎片会造成假 OOM（3780 实测）；torch 在 predict 时才导入，此处设置仍生效。
    os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
    try:
        print(run(argv))
        return 0
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        print(f'深度 GeoTIFF 生成失败：{exc}',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
