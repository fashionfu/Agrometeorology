@echo off
setlocal enabledelayedexpansion

REM 启用命令回显以便调试（如果需要可以注释掉）
REM @echo on

REM 设置代码页为 UTF-8
chcp 65001 >nul 2>&1

REM 设置错误处理标志
set "ERROR_OCCURRED=0"

echo ========================================
echo Push to LycheeFlowerFruitGrayMold folder
echo Repository: https://github.com/fashionfu/Agrometeorology.git
echo Only clear files in LycheeFlowerFruitGrayMold except README.md
echo ========================================
echo.
echo Press Ctrl+C to cancel at any time
echo.

REM 检查 Git 是否安装
echo [INFO] Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH!
    echo.
    echo Please install Git from: https://git-scm.com/downloads
    echo Or add Git to your system PATH
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('git --version 2^>nul') do set GIT_VERSION=%%v
echo [OK] Git found: %GIT_VERSION%
echo.

echo Starting script execution...
echo.

REM 切换到项目根目录
echo [INFO] Setting working directory...
cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Failed to change to script directory!
    pause
    exit /b 1
)
echo Current directory: %CD%
if not exist "%CD%" (
    echo [ERROR] Current directory does not exist!
    pause
    exit /b 1
)
echo.

REM 检查是否在 Git 仓库中
if not exist ".git" (
    echo [WARNING] Current directory is not a Git repository!
    echo.
    echo Checking parent directories for Git repository...
    set PARENT_CHECKED=0
    set ORIGINAL_DIR=%CD%
    cd ..
    if exist ".git" (
        echo Found Git repository in parent directory.
        echo Current directory: %CD%
        set PARENT_CHECKED=1
        echo.
    ) else (
        cd "%ORIGINAL_DIR%"
        echo [INFO] No Git repository found in parent directory.
        echo.
        echo Initializing Git repository in current directory...
        git init
        if errorlevel 1 (
            echo [ERROR] Failed to initialize Git repository!
            echo Please ensure Git is installed and accessible.
            echo.
            pause
            exit /b 1
        )
        echo [OK] Git repository initialized.
        echo.
        REM 配置远程仓库
        echo Configuring remote repository...
        git remote add origin https://github.com/fashionfu/Agrometeorology.git 2>nul
        if errorlevel 1 (
            echo [INFO] Remote may already exist, updating URL...
            git remote set-url origin https://github.com/fashionfu/Agrometeorology.git
            if errorlevel 1 (
                echo [WARNING] Failed to configure remote repository
            ) else (
                echo [OK] Remote repository URL updated
            )
        ) else (
            echo [OK] Remote repository added
        )
        echo.
        REM 注意：不在这里进行初始提交，让后续步骤处理文件
    )
)

REM 再次确认 Git 仓库
echo [INFO] Verifying Git repository...
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git repository check failed!
    echo.
    echo This may happen if:
    echo 1. Git is not installed
    echo 2. The .git folder is corrupted
    echo 3. You don't have permission to access the directory
    echo.
    echo Please check:
    echo - Is Git installed? Run: git --version
    echo - Is this a valid directory?
    echo.
    pause
    exit /b 1
)
echo [OK] Git repository verified.
echo.

REM 检查 Git 用户配置
echo [Step 0] Checking Git user configuration...
git config user.name >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Git user name not configured!
    echo Please configure Git user name and email:
    echo   git config --global user.name "Your Name"
    echo   git config --global user.email "your.email@example.com"
    echo.
    echo Attempting to continue, but commit may fail if not configured.
    echo.
)

REM 配置远程仓库
echo [Step 1] Configuring remote repository...
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo Adding remote repository...
    git remote add origin https://github.com/fashionfu/Agrometeorology.git
    if errorlevel 1 (
        echo [ERROR] Failed to add remote repository!
        pause
        exit /b 1
    )
    echo [OK] Remote repository added
) else (
    echo Checking remote URL...
    git remote set-url origin https://github.com/fashionfu/Agrometeorology.git
    echo [OK] Remote repository configured
)
git remote -v
echo.

REM 获取当前分支名
echo [Step 2] Getting current branch information...
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set CURRENT_BRANCH=%%i
if "%CURRENT_BRANCH%"=="" (
    for /f "tokens=2" %%i in ('git branch --show-current 2^>nul') do set CURRENT_BRANCH=%%i
)
if "%CURRENT_BRANCH%"=="" (
    echo [INFO] No branch found, checking for default branch...
    REM 尝试创建 main 分支
    git checkout -b main >nul 2>&1
    if not errorlevel 1 (
        set CURRENT_BRANCH=main
        echo [OK] Created and switched to main branch
    ) else (
        REM 如果 main 失败，尝试 master
        git checkout -b master >nul 2>&1
        if not errorlevel 1 (
            set CURRENT_BRANCH=master
            echo [OK] Created and switched to master branch
        ) else (
            set CURRENT_BRANCH=main
            echo [WARNING] Cannot create branch, will try default: main
        )
    )
)
echo Current branch: %CURRENT_BRANCH%
echo.

REM 获取远程分支信息
echo [Step 3] Fetching remote repository...
git fetch origin
if errorlevel 1 (
    echo [WARNING] Fetch failed, may be first push
)
echo.

REM 删除 LycheeFlowerFruitGrayMold 文件夹中除了 README.md 以外的所有已跟踪文件
echo [Step 4] Removing files in LycheeFlowerFruitGrayMold except README.md...
echo This will ONLY affect files in LycheeFlowerFruitGrayMold folder
echo Other folders in the repository will NOT be affected
echo.

REM 获取 LycheeFlowerFruitGrayMold 文件夹下所有已跟踪的文件列表
echo Getting list of tracked files in LycheeFlowerFruitGrayMold...
git ls-files "LycheeFlowerFruitGrayMold/" > temp_files.txt 2>nul

REM 删除除了 README.md 以外的所有文件（只处理 LycheeFlowerFruitGrayMold 文件夹）
if exist temp_files.txt (
    echo Processing files to remove...
    set REMOVED_COUNT=0
    for /f "tokens=*" %%f in (temp_files.txt) do (
        set "FILE_NAME=%%f"
        set "FILE_NAME=!FILE_NAME:\=/!"
        REM 检查是否是 LycheeFlowerFruitGrayMold/README.md
        echo !FILE_NAME! | findstr /i /c:"LycheeFlowerFruitGrayMold/README.md" >nul
        if errorlevel 1 (
            REM 不是 README.md，删除它（但只删除 LycheeFlowerFruitGrayMold 下的文件）
            echo !FILE_NAME! | findstr /i /c:"LycheeFlowerFruitGrayMold/" >nul
            if not errorlevel 1 (
                echo Removing: %%f
                git rm --cached "%%f" >nul 2>&1
                if not errorlevel 1 (
                    set /a REMOVED_COUNT+=1
                )
            )
        ) else (
            REM 是 LycheeFlowerFruitGrayMold/README.md，保留
            echo Keeping: %%f (README.md in LycheeFlowerFruitGrayMold)
        )
    )
    del temp_files.txt >nul 2>&1
    echo [OK] Removed %REMOVED_COUNT% files from Git index in LycheeFlowerFruitGrayMold (README.md preserved)
) else (
    echo [INFO] No tracked files found in LycheeFlowerFruitGrayMold or first time setup
)
echo.

REM 确保 LycheeFlowerFruitGrayMold 目录存在
if not exist "LycheeFlowerFruitGrayMold" (
    echo Creating LycheeFlowerFruitGrayMold directory...
    mkdir LycheeFlowerFruitGrayMold
)

REM 将当前目录的文件和文件夹复制到 LycheeFlowerFruitGrayMold（排除 .git、LycheeFlowerFruitGrayMold 和脚本本身）
echo [Step 5] Copying current directory to LycheeFlowerFruitGrayMold...
echo Excluding: .git, LycheeFlowerFruitGrayMold, 1121推送.bat
echo.

REM 复制文件
for %%f in (*.*) do (
    if /i not "%%f"=="1121推送.bat" (
        if exist "%%f" (
            echo Copying file: %%f
            copy /Y "%%f" "LycheeFlowerFruitGrayMold\%%f" >nul 2>&1
        )
    )
)

REM 复制子目录（排除 .git 和 LycheeFlowerFruitGrayMold）
for /d %%d in (*) do (
    if /i not "%%d"==".git" if /i not "%%d"=="LycheeFlowerFruitGrayMold" (
        if exist "%%d" (
            echo Copying directory: %%d
            xcopy /E /I /Y /Q "%%d" "LycheeFlowerFruitGrayMold\%%d\" >nul 2>&1
        )
    )
)
echo [OK] Files copied to LycheeFlowerFruitGrayMold
echo.

REM 添加 LycheeFlowerFruitGrayMold 文件夹到 Git
echo [Step 6] Adding LycheeFlowerFruitGrayMold folder to Git...
git add LycheeFlowerFruitGrayMold/
if errorlevel 1 (
    echo [ERROR] Failed to add files!
    pause
    exit /b 1
)
echo [OK] Files added to staging area
echo.

REM 检查是否有需要提交的更改
echo [Step 7] Checking changes to commit...
git diff --cached --quiet
if errorlevel 1 (
    echo Changes detected, committing...
    REM 检查是否是首次提交
    git rev-parse --verify HEAD >nul 2>&1
    if errorlevel 1 (
        echo This is the first commit.
        git commit -m "Initial commit: Add LycheeFlowerFruitGrayMold project"
    ) else (
        git commit -m "Update LycheeFlowerFruitGrayMold: clear old files and push entire project folder"
    )
    if errorlevel 1 (
        echo [ERROR] Commit failed!
        echo.
        echo If this is your first commit, you may need to configure Git user:
        echo   git config --global user.name "Your Name"
        echo   git config --global user.email "your.email@example.com"
        pause
        exit /b 1
    )
    echo [OK] Changes committed
) else (
    echo [INFO] No changes to commit
)
echo.

REM 尝试推送方法 1: git push origin <当前分支>
echo ========================================
echo [Method 1] Trying: git push origin %CURRENT_BRANCH%
echo ========================================
git push origin %CURRENT_BRANCH%
if not errorlevel 1 (
    echo.
    echo [SUCCESS] Push successful!
    echo Completed using Method 1
    pause
    exit /b 0
)
echo [FAILED] Method 1 push failed
echo.

REM 尝试推送方法 2: git push -u origin <当前分支> (首次推送)
echo ========================================
echo [Method 2] Trying: git push -u origin %CURRENT_BRANCH%
echo ========================================
git push -u origin %CURRENT_BRANCH%
if not errorlevel 1 (
    echo.
    echo [SUCCESS] Push successful!
    echo Completed using Method 2
    pause
    exit /b 0
)
echo [FAILED] Method 2 push failed
echo.

REM 尝试推送方法 3: git push origin HEAD
echo ========================================
echo [Method 3] Trying: git push origin HEAD
echo ========================================
git push origin HEAD
if not errorlevel 1 (
    echo.
    echo [SUCCESS] Push successful!
    echo Completed using Method 3
    pause
    exit /b 0
)
echo [FAILED] Method 3 push failed
echo.

REM 尝试推送方法 4: git push origin main (强制尝试 main)
echo ========================================
echo [Method 4] Trying: git push origin main
echo ========================================
git push origin main
if not errorlevel 1 (
    echo.
    echo [SUCCESS] Push successful!
    echo Completed using Method 4
    pause
    exit /b 0
)
echo [FAILED] Method 4 push failed
echo.

REM 尝试推送方法 5: git push origin master (强制尝试 master)
echo ========================================
echo [Method 5] Trying: git push origin master
echo ========================================
git push origin master
if not errorlevel 1 (
    echo.
    echo [SUCCESS] Push successful!
    echo Completed using Method 5
    pause
    exit /b 0
)
echo [FAILED] Method 5 push failed
echo.

REM 所有方法都失败
echo ========================================
echo [ERROR] All push methods failed!
echo ========================================
echo.
echo Possible reasons:
echo 1. Network connection issue
echo 2. GitHub authentication failed (need Personal Access Token)
echo 3. Remote repository permission denied
echo 4. Branch name mismatch
echo.
echo Suggested actions:
echo 1. Check network connection
echo 2. Verify GitHub authentication
echo 3. Manually execute: git push origin %CURRENT_BRANCH%
echo 4. Check detailed error messages
echo.
echo ========================================
echo Script execution completed with errors
echo ========================================
echo.
pause
exit /b 1

REM 通用错误处理（如果脚本意外退出）
:ERROR_HANDLER
echo.
echo ========================================
echo [FATAL ERROR] Script encountered an unexpected error!
echo ========================================
echo.
echo Please check the error messages above.
echo.
echo If the problem persists, try:
echo 1. Running the script from Command Prompt to see detailed errors
echo 2. Checking Git installation: git --version
echo 3. Verifying network connection
echo.
pause
exit /b 1
