#!/bin/bash

echo "=========================================="
echo "🚀 RAG MVP 系统启动脚本"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "📌 Python版本: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "📚 安装依赖包..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 检查端口是否被占用
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  端口8000已被占用，尝试关闭..."
    kill $(lsof -Pi :8000 -sTCP:LISTEN -t)
    sleep 2
fi

# 启动后端服务
echo "=========================================="
echo "🎯 启动后端API服务..."
echo "=========================================="
python3 api/main.py &
API_PID=$!

# 等待API启动
echo "⏳ 等待API服务启动..."
sleep 5

# 检查API是否启动成功
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "✅ API服务启动成功！"
else
    echo "❌ API服务启动失败"
    exit 1
fi

# 启动前端（使用Python简单服务器）
echo "=========================================="
echo "🌐 启动前端服务..."
echo "=========================================="
python3 -m http.server 3000 --directory . > /dev/null 2>&1 &
WEB_PID=$!

echo ""
echo "=========================================="
echo "✨ RAG MVP系统启动成功！"
echo "=========================================="
echo ""
echo "📍 访问地址:"
echo "   - 前端界面: http://localhost:3000"
echo "   - API文档: http://localhost:8000/docs"
echo "   - 健康检查: http://localhost:8000/api/health"
echo ""
echo "📌 进程信息:"
echo "   - API进程: $API_PID"
echo "   - Web进程: $WEB_PID"
echo ""
echo "⌨️  按 Ctrl+C 停止所有服务"
echo "=========================================="

# 捕获退出信号
trap "echo '正在停止服务...'; kill $API_PID $WEB_PID; exit" INT TERM

# 保持脚本运行
wait