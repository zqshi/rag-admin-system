#!/bin/bash

echo "请输入你的GitHub仓库URL (例如: https://github.com/username/repo-name.git):"
read REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "错误：仓库URL不能为空"
    exit 1
fi

echo "添加远程仓库..."
git remote add origin "$REPO_URL"

echo "推送到远程仓库..."
git push -u origin main

echo "✅ 推送完成！"
echo ""
echo "仓库已成功推送到: $REPO_URL"
echo ""
echo "你可以访问以下链接查看:"
echo "${REPO_URL%.git}"