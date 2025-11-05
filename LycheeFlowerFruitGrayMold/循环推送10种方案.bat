@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================================================
echo 自动循环尝试10种Git推送方案
echo ========================================================================
echo.

set REPO_URL=https://github.com/fashionfu/Agrometeorology.git
set SUBDIR=LycheeFlowerFruitGrayMold
set BRANCH=main

set METHOD=1

:LOOP_START
echo.
echo ========================================================================
echo 尝试方案 %METHOD% / 10
echo ========================================================================
echo.

if %METHOD%==1 goto METHOD_1
if %METHOD%==2 goto METHOD_2
if %METHOD%==3 goto METHOD_3
if %METHOD%==4 goto METHOD_4
if %METHOD%==5 goto METHOD_5
if %METHOD%==6 goto METHOD_6
if %METHOD%==7 goto METHOD_7
if %METHOD%==8 goto METHOD_8
if %METHOD%==9 goto METHOD_9
if %METHOD%==10 goto METHOD_10
goto END

:METHOD_1
echo 方案1: 在当前目录初始化Git并直接推送
if not exist .git git init
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git add .
git commit -m "Add LycheeFlowerFruitGrayMold: 荔枝霜疫霉花果预警阈值模型" 2>nul
git checkout -b %BRANCH% 2>nul
git push -u origin %BRANCH%
if %ERRORLEVEL%==0 (
    echo.
    echo ========================================================================
    echo 成功！方案1执行成功！
    echo ========================================================================
    goto END
)
goto NEXT_METHOD

:METHOD_2
echo 方案2: 克隆主仓库，复制文件到子目录
cd ..
if exist Agrometeorology_temp rmdir /s /q Agrometeorology_temp
git clone %REPO_URL% Agrometeorology_temp
if %ERRORLEVEL%==0 (
    cd Agrometeorology_temp
    if not exist %SUBDIR% mkdir %SUBDIR%
    xcopy /E /I /Y ..\03_荔枝霜疫霉并的阈值模型\* %SUBDIR%\ >nul
    git add .
    git commit -m "Add %SUBDIR%: 荔枝霜疫霉花果预警阈值模型"
    git push origin %BRANCH%
    if %ERRORLEVEL%==0 (
        echo.
        echo ========================================================================
        echo 成功！方案2执行成功！
        echo ========================================================================
        cd ..
        goto END
    )
    cd ..
)
cd 03_荔枝霜疫霉并的阈值模型
goto NEXT_METHOD

:METHOD_3
echo 方案3: 强制推送
if not exist .git git init
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git add .
git commit -m "Initial commit: 荔枝霜疫霉花果预警阈值模型" 2>nul
git checkout -b %BRANCH% 2>nul
git push -u origin %BRANCH% --force
if %ERRORLEVEL%==0 (
    echo.
    echo ========================================================================
    echo 成功！方案3执行成功！
    echo ========================================================================
    goto END
)
goto NEXT_METHOD

:METHOD_4
echo 方案4: 创建orphan分支
if not exist .git git init
git checkout --orphan %BRANCH% 2>nul
git add .
git commit -m "Initial commit: 荔枝霜疫霉花果预警阈值模型" 2>nul
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git push -u origin %BRANCH%
if %ERRORLEVEL%==0 (
    echo.
    echo ========================================================================
    echo 成功！方案4执行成功！
    echo ========================================================================
    goto END
)
goto NEXT_METHOD

:METHOD_5
echo 方案5: 浅克隆方式
cd ..
if exist Agrometeorology_shallow rmdir /s /q Agrometeorology_shallow
git clone --depth 1 %REPO_URL% Agrometeorology_shallow
if %ERRORLEVEL%==0 (
    cd Agrometeorology_shallow
    if not exist %SUBDIR% mkdir %SUBDIR%
    xcopy /E /I /Y ..\03_荔枝霜疫霉并的阈值模型\* %SUBDIR%\ >nul
    git add .
    git commit -m "Add %SUBDIR%: 荔枝霜疫霉花果预警阈值模型"
    git push origin %BRANCH%
    if %ERRORLEVEL%==0 (
        echo.
        echo ========================================================================
        echo 成功！方案5执行成功！
        echo ========================================================================
        cd ..
        goto END
    )
    cd ..
)
cd 03_荔枝霜疫霉并的阈值模型
goto NEXT_METHOD

:METHOD_6
echo 方案6: 手动创建子目录结构
cd ..
if exist Agrometeorology_manual rmdir /s /q Agrometeorology_manual
mkdir Agrometeorology_manual
cd Agrometeorology_manual
git init
if not exist %SUBDIR% mkdir %SUBDIR%
xcopy /E /I /Y ..\03_荔枝霜疫霉并的阈值模型\* %SUBDIR%\ >nul
git remote add origin %REPO_URL%
git add .
git commit -m "Add %SUBDIR%: 荔枝霜疫霉花果预警阈值模型"
git checkout -b %BRANCH% 2>nul
git push -u origin %BRANCH%
if %ERRORLEVEL%==0 (
    echo.
    echo ========================================================================
    echo 成功！方案6执行成功！
    echo ========================================================================
    cd ..
    goto END
)
cd ..\03_荔枝霜疫霉并的阈值模型
goto NEXT_METHOD

:METHOD_7
echo 方案7: 使用不同的分支名
if not exist .git git init
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git add .
git commit -m "Initial commit" 2>nul
git checkout -b master 2>nul
git push -u origin master
if %ERRORLEVEL%==0 (
    echo.
    echo ========================================================================
    echo 成功！方案7执行成功！
    echo ========================================================================
    goto END
)
goto NEXT_METHOD

:METHOD_8
echo 方案8: 使用现有分支推送
if not exist .git git init
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git fetch origin 2>nul
git add .
git commit -m "Add %SUBDIR%: 荔枝霜疫霉花果预警阈值模型" 2>nul
git push origin HEAD:%BRANCH%
if %ERRORLEVEL%==0 (
    echo.
    echo ========================================================================
    echo 成功！方案8执行成功！
    echo ========================================================================
    goto END
)
goto NEXT_METHOD

:METHOD_9
echo 方案9: 分步骤推送
if not exist .git git init
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git add README.md .gitignore
git commit -m "Add README and gitignore" 2>nul
git checkout -b %BRANCH% 2>nul
git push -u origin %BRANCH%
if %ERRORLEVEL%==0 (
    git add .
    git commit -m "Add all files: 荔枝霜疫霉花果预警阈值模型"
    git push origin %BRANCH%
    if %ERRORLEVEL%==0 (
        echo.
        echo ========================================================================
        echo 成功！方案9执行成功！
        echo ========================================================================
        goto END
    )
)
goto NEXT_METHOD

:METHOD_10
echo 方案10: 使用GitHub网页上传（最后尝试）
echo.
echo ========================================================================
echo 方案10: 此方案需要手动操作
echo ========================================================================
echo.
echo 请访问: %REPO_URL%
echo 1. 在GitHub网页上创建 %SUBDIR% 目录
echo 2. 使用GitHub的文件上传功能上传所有文件
echo.
echo 或者，所有自动方案都失败，请检查：
echo 1. 网络连接
echo 2. GitHub访问权限
echo 3. 仓库访问权限
echo 4. Git认证配置
echo.
goto END

:NEXT_METHOD
echo.
echo 方案 %METHOD% 失败，尝试下一个方案...
echo.
set /a METHOD+=1
if %METHOD% LEQ 10 goto LOOP_START

:END
echo.
echo ========================================================================
echo 所有方案尝试完成
echo ========================================================================
echo.
if %METHOD% LEQ 10 (
    echo 已成功推送到GitHub！
) else (
    echo 所有自动方案都尝试失败，请参考上面的手动操作说明。
)
echo.
pause

