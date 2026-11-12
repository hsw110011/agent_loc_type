# Skill：高程结构评估步骤

1. **定位目标路面**：用第 1 张带黄框的 RGB 上下文查看道路走向、地形和可能的桥面或遮挡。回到第 2 张局部 RGB，在中心附近约 10 m 辨认道路、广场或场院，按 Rule 填写 `drivable_check`。
2. **核对数据**：检查第 2、3 张关键位置是否对齐，白色无数据是否覆盖路面或路侧结构，按 Rule 填写 `data_check`。把疑似异常写成有图像依据的观察。
3. **比较同位置的相对高度**：先从第 2 张找到可辨路面，在第 3 张读取该位置的颜色，再比较路侧地物的高低、范围和轮廓。第 3 张着色的是 DSM 表面相对高程，路面未必落在色带最低端。同色可能来自色带端点饱和，白色缺失不计作低处。在 `height_comparisons.road_and_surroundings` 回答路面颜色、路侧高出结构的位置、轮廓和分布。
4. **查沿路区分度与遮挡**：用 RGB 判断道路几何，在 DSM 沿路比较至少两处两侧的高低轮廓、间距与分布。将沿路比较写入 `height_comparisons.along_road`。在 `height_comparisons.overhead_occlusion` 写遮挡位置和 DSM 表面特征，或说明未见迹象、无法判断。关键结构看不清时，按 Rule 填写相应的不确定性。
5. **输出观察**：逐图写 `observations`，在 `height_check.evidence` 指出支持结构判断的位置与证据，按 Rule 填各字段及其置信度。最后填写 `verdict.result`、`verdict.reason` 和 `verdict.evidence_refs`，引用支撑结论的对照位置。

6. **核对后再提交结论**：结构不足引用 `height_comparisons.road_and_surroundings`；纵向退化须先在 `longitudinal_comparisons` 分别记录两处高度轮廓。不退化须有高出结构充分及可排除纵向退化的证据。俯视遮挡只用于说明潜在失配。 先填 `verdict.evidence_refs`，编号按 Rule。 `verdict.reason` 只能引用前面已记录、能在图中定位的证据。保留“直接可见”“有依据的推断”和“无法判断”的区别；不补写未观察到的地物、数值或车端 BEV 内容。证据互相矛盾或关键证据不足，且其余证据无法独立满足 Rule 时，选「证据不足」并说明缺项。
