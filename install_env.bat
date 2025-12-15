@echo off
setlocal enabledelayedexpansion

:: 0. 升级 pip 以避免安装时的构建错误
echo [0/4] Upgrading pip...
pip install --upgrade pip

:: 1. 安装基础依赖
echo [1/4] Installing base requirements...
pip install -r requirements.txt

:: 2. 安装 PyTorch 2.3.0 全家桶 (CPU版)
:: 注意：torchvision 对应 0.18.0
echo [2/4] Installing PyTorch 2.3.0 (CPU)...
pip install torch==2.3.0+cpu ^
            torchvision==0.18.0+cpu ^
            torchaudio==2.3.0+cpu ^
            --index-url https://download.pytorch.org/whl/cpu

:: 3. 安装 PyG (Graph) 依赖
:: 必须使用 -f 指定针对 torch-2.3.0 的 CPU 预编译包，否则会失败
echo [3/4] Installing PyG dependencies...
pip install torch-scatter==2.1.2 ^
            torch-sparse==0.6.18 ^
            -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

:: 安装 PyG 主库和 torchdata (指定了 0.7.1)
pip install torch-geometric==2.5.3 torchdata==0.7.1

:: 4. 安装 DGL 2.2.1
:: 指定适配 torch-2.3 的源，确保 graphbolt 兼容
echo [4/4] Installing DGL 2.2.1...
pip install dgl==2.2.1 ^
            -f https://data.dgl.ai/wheels/torch-2.3/repo.html

echo === ✅ Environment Setup Complete! ===
:: 验证安装
python -c "import torch; import dgl; print(f'Torch: {torch.__version__}, DGL: {dgl.__version__}')"

endlocal