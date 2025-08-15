#!/bin/bash

echo "🚀 启动RAG智能问答系统..."
echo "================================"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行："
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# 激活虚拟环境
echo "📦 激活虚拟环境..."
source venv/bin/activate

# 检查依赖
echo "🔍 检查依赖..."
pip list | grep -q "fastapi" || {
    echo "❌ 缺少依赖，请运行: pip install -r requirements.txt"
    exit 1
}

# 启动后端API服务
echo "🔧 启动后端API服务 (端口8000)..."
python api/main.py &
API_PID=$!

# 等待API服务启动
echo "⏳ 等待API服务启动..."
sleep 5

# 检查API是否启动成功
if curl -s http://localhost:8000/api/statistics > /dev/null; then
    echo "✅ API服务启动成功"
else
    echo "❌ API服务启动失败"
    kill $API_PID 2>/dev/null
    exit 1
fi

# 启动前端HTTP服务
echo "🌐 启动前端HTTP服务 (端口8080)..."
python3 -m http.server 8080 &
WEB_PID=$!

# 等待前端服务启动
sleep 2

echo ""
echo "🎉 系统启动完成！"
echo "================================"
echo "📍 访问地址："
echo "   🌐 Web界面:    http://localhost:8080"
echo "   📚 API文档:    http://localhost:8000/docs" 
echo "   💊 健康检查:   http://localhost:8000/api/health"
echo ""
echo "💡 功能特性："
echo "   📄 支持文档上传 (PDF/TXT/MD/DOC)"
echo "   🔍 智能语义搜索"
echo "   💬 AI问答对话"
echo "   📊 实时统计监控"
echo ""
echo "⚡ 快速测试："
echo "   1. 打开 http://localhost:8080"
echo "   2. 上传一个文档"
echo "   3. 输入问题进行智能问答"
echo ""
echo "🛑 停止服务: 按 Ctrl+C"

# 等待用户中断
trap 'echo ""; echo "🛑 正在停止服务..."; kill $API_PID $WEB_PID 2>/dev/null; echo "✅ 服务已停止"; exit 0' INT

# 保持脚本运行
wait