# GitHub 发布

本仓库可按实验项目初版发布。README 和结果页已区分历史实测与代码整理后的验证范围；尚未完成 GPU 回归的入口不标作已验收。模型权重、完整数据集、运行日志、个人简历及大型 trace 不纳入 Git。

仓库名为 `infer-perf-lab`，由 GitHub 用户 `mocaoo` 维护。代码许可尚未指定；没有 LICENSE 时不应自称已经提供某种开源许可证。

## 上传代码

在 GitHub 创建空仓库后，从本地仓库根目录执行；将远端地址替换为自己实际创建的仓库。Git 提交身份使用自己的 GitHub 提交邮箱或 noreply 邮箱。

```bash
git add .
git commit -m "Organize single-GPU inference experiments and results"
git remote add origin <your-repository-git-url>
git push -u origin main
```

也可将提供的压缩包解压后，通过 GitHub 网页上传。首次推送后检查 CPU checks 工作流；它验证客户端和静态检查，不运行 GPU。

## 展示 HTML

`docs/index.html` 是可独立打开的单文件，16 张图以 SVG data URI 内嵌。`docs/experiments.md` 使用仓库相对 PNG 路径，可在 GitHub 文件页直接阅读。

需要网页预览时，在仓库 Settings → Pages 中选择从分支部署，选择 `main` 和 `/docs`，保存后使用 GitHub 返回的站点地址。以仓库实际部署状态及返回地址为准。

后续更新图文时，同时更新 `docs/index.html`、`docs/experiments.md`、`docs/assets/` 和 `docs/gallery-provenance.json`，保留结果日期及实验版本范围。
