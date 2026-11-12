"""独立深度/语义工具 CLI：单图或 manifest 的指定块，不接入道路主流程。"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
from PIL import Image

from agent.paths import relative_path,resolve_path,write_json
from agent.perception_config import load_perception_config
from percection_tool.factory import create_depth_estimator,create_semantic_segmenter


def run(argv: list[str] | None = None) -> Path:
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--image')
    inputs.add_argument('--manifest')
    parser.add_argument('--block-id')
    parser.add_argument('--tool',choices=['depth','semantic','both'],default='both')
    parser.add_argument('--config',default='agent/perception_tools.yaml')
    parser.add_argument('--output-dir',default='output/perception_tools')
    args=parser.parse_args(argv)
    if bool(args.manifest) != bool(args.block_id):
        parser.error('--manifest 和 --block-id 必须同时提供')
    config=load_perception_config(args.config)
    limits=[cfg.max_input_pixels for name,cfg in [('depth',config.depth),('semantic',config.pipeline_semantic_config)]
            if args.tool in {name,'both'}]
    maximum=min(limits)
    # 文件读取由 Agent 处理，感知适配器仅接收 RGB 数组，不依赖 LangGraph。
    from agent.map_preprocess.io import open_image,load_manifest
    origin={}
    if args.image:
        with open_image(args.image,maximum) as source:
            if source.width*source.height>maximum:
                raise ValueError('图像超过工具 max_input_pixels；请用 --manifest/--block-id 按需读取')
            bbox=[0,0,source.width,source.height]
            rgb,valid=source.read(bbox),source.valid_mask(bbox)
        name=Path(args.image).stem
        origin={'image':relative_path(args.image)}
    else:
        from agent.map_preprocess import crop_block
        manifest=load_manifest(args.manifest)
        matches=[b for b in manifest['blocks'] if b['block_id']==args.block_id]
        if len(matches)!=1:
            raise ValueError(f'块 ID 不存在或不唯一：{args.block_id}')
        block=matches[0]
        if block['size_px']**2>maximum:
            raise ValueError('块超过工具 max_input_pixels')
        rgb=crop_block(manifest,args.block_id)
        valid=np.zeros(rgb.shape[:2],bool)
        x0,y0,x1,y1=block['valid_window_bbox']
        if x1>x0 and y1>y0:
            with open_image(manifest['image']['path'],manifest['config']['pillow_max_image_pixels']) as source:
                valid[y0:y1,x0:x1]=source.valid_mask(block['valid_global_bbox'])
        name=f"{Path(manifest['image']['path']).stem}_{args.block_id}"
        origin={'manifest':relative_path(args.manifest),'block_id':args.block_id,'global_xy':block['global_xy'],
                'crop_bbox':block['crop_bbox'],'resolution_m_per_px':manifest['resolution_m_per_px']}
    directory=resolve_path(args.output_dir)/(name.strip('. ') or 'image')/uuid4().hex
    directory.mkdir(parents=True,exist_ok=False)
    Image.fromarray(rgb).save(directory/'input.png')
    Image.fromarray(valid.astype(np.uint8)*255).save(directory/'valid_mask.png')
    write_json(directory/'config.json',config.to_dict())
    result={'source':origin,'config':config.to_dict(),'files':{'input':relative_path(directory/'input.png'),
            'valid_mask':relative_path(directory/'valid_mask.png')},'tools':{}}
    try:
        for name,cfg,creator in [('depth',config.depth,create_depth_estimator),('semantic',config.pipeline_semantic_config,create_semantic_segmenter)]:
            if args.tool not in {name,'both'}:
                continue
            tool=creator(cfg)
            try:
                output=tool.predict(rgb,valid_mask=valid)
            finally:
                tool.close()
            result['tools'][name]=output['metadata']
            if name=='depth':
                values=output['depth']
                np.save(directory/'depth.npy',values,allow_pickle=False)
                finite=np.isfinite(values)
                preview=np.zeros(values.shape,np.uint8)
                if finite.any():
                    low,high=float(values[finite].min()),float(values[finite].max())
                    if high>low:
                        preview[finite]=np.rint((values[finite]-low)/(high-low)*255).astype(np.uint8)
                Image.fromarray(preview).save(directory/'depth_preview.png')
                result['files'].update({key:relative_path(directory/file) for key,file in
                    [('depth','depth.npy'),('depth_preview','depth_preview.png')]})
            else:
                mask=output['mask']
                np.save(directory/'semantic_confidence.npy',output['confidence'],allow_pickle=False)
                color=np.full(rgb.shape,64,dtype=np.uint8)
                for index,rgb_color in enumerate(output['metadata']['palette_rgb']):
                    color[mask==index]=rgb_color
                overlay=rgb.copy()
                overlay[valid]=np.rint(.5*rgb[valid].astype(np.float32)+.5*color[valid]).astype(np.uint8)
                for filename,array in [('semantic_mask.png',mask),('semantic_color.png',color),('semantic_overlay.png',overlay)]:
                    Image.fromarray(array).save(directory/filename)
                    result['files'][filename[:-4]]=relative_path(directory/filename)
                result['files']['semantic_confidence']=relative_path(directory/'semantic_confidence.npy')
        write_json(directory/'result.json',result)
    except Exception as exc:
        write_json(directory/'error.json',{'error':str(exc),'completed_tools':list(result['tools'])})
        raise
    return directory/'result.json'


def main(argv: list[str] | None = None) -> int:
    try:
        print(run(argv))
        return 0
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        print(f'分析工具执行失败：{exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
