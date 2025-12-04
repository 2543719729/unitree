# ============================================
# Git Submodules 配置脚本
# 用途：将所有子项目配置为 submodules
# ============================================

$GITHUB_USERNAME = "2543719729"  # 你的 GitHub 用户名

Write-Host "======================================"
Write-Host "配置 Git Submodules"
Write-Host "======================================"

# 1. 更新 .gitignore - 移除子项目忽略
Write-Host "`n[1/6] 更新 .gitignore..."
@"
.idea/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/

# 训练日志和模型
logs/
*.pt
*.pth

# IDE
.vscode/
*.swp
*.swo
"@ | Out-File -FilePath .gitignore -Encoding utf8

# 2. 配置 unitree_rl_lab
Write-Host "`n[2/6] 配置 unitree_rl_lab..."
cd unitree_rl_lab
if (git remote get-url origin 2>$null) {
    Write-Host "origin 已存在，跳过"
} else {
    git remote add origin "https://github.com/$GITHUB_USERNAME/unitree_rl_lab.git"
    Write-Host "已添加 origin: $GITHUB_USERNAME/unitree_rl_lab"
}
# 推送代码
Write-Host "推送 unitree_rl_lab..."
git push -u origin main
cd ..

# 3. 配置 unitree_mujoco
Write-Host "`n[3/6] 配置 unitree_mujoco..."
cd unitree_mujoco
if (git remote get-url upstream 2>$null) {
    Write-Host "upstream 已存在，跳过"
} else {
    git remote rename origin upstream
    git remote add origin "https://github.com/$GITHUB_USERNAME/unitree_mujoco.git"
    Write-Host "已添加 origin: $GITHUB_USERNAME/unitree_mujoco"
}
# 提交并推送
if (Test-Path "g1_rl_train") {
    git add g1_rl_train
    git commit -m "添加 g1_rl_train 配置"
}
git push -u origin main
cd ..

# 4. unitree_ros 和 IsaacLab 保持官方
Write-Host "`n[4/6] unitree_ros 和 IsaacLab 使用官方仓库"

# 5. 移除现有子项目，准备添加为 submodules
Write-Host "`n[5/6] 准备添加为 submodules..."
git rm --cached -r unitree_rl_lab unitree_mujoco unitree_ros IsaacLab 2>$null

# 6. 添加为 submodules
Write-Host "`n[6/6] 添加 submodules..."

# unitree_rl_lab (使用你的 fork)
git submodule add "https://github.com/$GITHUB_USERNAME/unitree_rl_lab.git" unitree_rl_lab

# unitree_mujoco (使用你的 fork)
git submodule add "https://github.com/$GITHUB_USERNAME/unitree_mujoco.git" unitree_mujoco

# unitree_ros (使用官方仓库)
git submodule add "https://github.com/unitreerobotics/unitree_ros.git" unitree_ros

# IsaacLab (使用官方仓库)
git submodule add "https://github.com/isaac-sim/IsaacLab.git" IsaacLab

# 7. 提交配置
Write-Host "`n提交 submodule 配置..."
git add .gitmodules .gitignore TRAINING_CONFIG_GUIDE.md
git commit -m "配置所有子项目为 submodules"

# 8. 推送到远程
Write-Host "`n推送到远程仓库..."
git push unitree stair_backup

Write-Host "`n======================================"
Write-Host "✅ 配置完成！"
Write-Host "======================================"
Write-Host ""
Write-Host "云服务器部署命令："
Write-Host "  git clone --recursive https://github.com/$GITHUB_USERNAME/unitree.git"
Write-Host ""
