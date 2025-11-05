@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================================================
echo 循环尝试10种方案推送文件到GitHub子目录
echo ========================================================================
echo.

set SUBDIR=LycheeFlowerFruitGrayMold
set METHOD=1
set MAX_METHOD=10

:MAIN_LOOP
echo.
echo ========================================================================
echo 尝试方案 %METHOD% / %MAX_METHOD%
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
echo 方案1: 排除Agrometeorology和.idea，添加并推送
if exist Agrometeorology\.git rmdir /s /q Agrometeorology\.git 2>nul
if exist .idea\.git rmdir /s /q .idea\.git 2>nul
git add . --ignore-errors 2>nul
git add %SUBDIR%/ 2>nul
git add scripts/ 2>nul
git add metadata/ 2>nul
git add analysis_1104/ 2>nul
git add analysis_1104_batch/ 2>nul
git add analysis_1105/ 2>nul
git add *.md 2>nul
git add *.bat 2>nul
git add *.py 2>nul
git add .gitignore 2>nul
git commit -m "Reorganize: Move all files to LycheeFlowerFruitGrayMold subdirectory" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_2
echo 方案2: 删除Agrometeorology目录后推送
if exist Agrometeorology rmdir /s /q Agrometeorology 2>nul
if exist .idea rmdir /s /q .idea 2>nul
git add . --ignore-errors 2>nul
git commit -m "Remove nested repos and reorganize" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_3
echo 方案3: 只添加子目录内容
git add %SUBDIR%/
git commit -m "Add LycheeFlowerFruitGrayMold subdirectory" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_4
echo 方案4: 使用.gitignore排除问题目录
echo Agrometeorology/ >> .gitignore 2>nul
echo .idea/ >> .gitignore 2>nul
echo __pycache__/ >> .gitignore 2>nul
echo ~$*.xlsx >> .gitignore 2>nul
echo ~$*.xls >> .gitignore 2>nul
git add .gitignore
git add .
git commit -m "Add .gitignore and reorganize files" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_5
echo 方案5: 强制添加所有文件
git add -f .
git add -f %SUBDIR%/
git commit -m "Force add all files" 2>nul
git push origin main --force
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_6
echo 方案6: 分步添加和推送
git add %SUBDIR%/README.md 2>nul
git add %SUBDIR%/scripts/ 2>nul
git commit -m "Add scripts" 2>nul
git push origin main 2>nul
if %ERRORLEVEL%==0 (
    git add %SUBDIR%/metadata/ 2>nul
    git add %SUBDIR%/analysis_*/ 2>nul
    git commit -m "Add data and analysis" 2>nul
    git push origin main
    if %ERRORLEVEL%==0 goto SUCCESS
)
goto NEXT_METHOD

:METHOD_7
echo 方案7: 清理后重新添加
git reset . 2>nul
if exist Agrometeorology\.git attrib -r /s /d Agrometeorology\.git 2>nul
git add %SUBDIR%/ --ignore-errors
git add .gitignore 2>nul
git commit -m "Reorganize files to subdirectory" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_8
echo 方案8: 使用不同分支名
git checkout -b master 2>nul
git add %SUBDIR%/ --ignore-errors
git commit -m "Reorganize files" 2>nul
git push origin master 2>nul
if %ERRORLEVEL%==0 (
    git checkout main 2>nul
    git merge master 2>nul
    git push origin main
    if %ERRORLEVEL%==0 goto SUCCESS
)
git checkout main 2>nul
goto NEXT_METHOD

:METHOD_9
echo 方案9: 直接推送子目录内容
cd %SUBDIR%
git init 2>nul
git add . 2>nul
git commit -m "Initial commit" 2>nul
git remote add origin https://github.com/fashionfu/Agrometeorology.git 2>nul
git push origin main:%SUBDIR% 2>nul
cd ..
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_10
echo 方案10: 最后尝试 - 清理所有问题后推送
if exist Agrometeorology rmdir /s /q Agrometeorology 2>nul
if exist .idea rmdir /s /q .idea 2>nul
for /d %%d in (*) do (
    if exist "%%d\.git" rmdir /s /q "%%d\.git" 2>nul
)
git add . --ignore-errors 2>nul
git commit -m "Final attempt: Reorganize to subdirectory" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:NEXT_METHOD
echo.
echo 方案 %METHOD% 失败，尝试下一个方案...
set /a METHOD+=1
if %METHOD% LEQ %MAX_METHOD% goto MAIN_LOOP
goto FAILED

:SUCCESS
echo.
echo ========================================================================
echo 成功！方案 %METHOD% 执行成功！
echo ========================================================================
echo.
echo 访问: https://github.com/fashionfu/Agrometeorology/tree/main/%SUBDIR%
echo.
goto END

:FAILED
echo.
echo ========================================================================
echo 所有方案都尝试失败
echo ========================================================================
echo.
echo 请手动执行以下命令:
echo 1. 删除问题目录: rmdir /s /q Agrometeorology
echo 2. 添加文件: git add %SUBDIR%/
echo 3. 提交: git commit -m "Reorganize files"
echo 4. 推送: git push origin main
echo.

:END
pause

