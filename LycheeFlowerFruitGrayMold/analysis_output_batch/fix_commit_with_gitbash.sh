#!/bin/bash
# 使用 Git Bash 修复提交信息（推荐方法）
# 这个脚本应该在 Git Bash 中运行

cd /e/temp_agrometeo_repo || {
    echo "克隆仓库..."
    git clone https://github.com/fashionfu/Agrometeorology.git /e/temp_agrometeo_repo
    cd /e/temp_agrometeo_repo
}

# 设置编码
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8

# 创建提交信息文件
cat > commit_msg.txt << 'EOF'
更新: 清空并重新上传LycheeFlowerFruitGrayMold文件夹内容 (2025-11-06)
EOF

# 修改提交
git commit --amend -F commit_msg.txt

# 显示提交信息
echo ""
echo "提交信息已修改："
git log -1 --pretty=format:"%s"

# 清理
rm commit_msg.txt

# 询问是否推送
echo ""
read -p "是否推送到远程仓库？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main --force
    echo "推送完成！请在 GitHub 网页上查看提交信息。"
else
    echo "请稍后手动推送: git push origin main --force"
fi

