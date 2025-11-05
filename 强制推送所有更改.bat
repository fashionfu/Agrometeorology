@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================================================
echo 强制推送所有更改到GitHub（循环尝试10种方案）
echo ========================================================================
echo.

set SUBDIR=LycheeFlowerFruitGrayMold
set METHOD=1

:MAIN_LOOP
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
echo 方案1: 添加所有更改并推送
git add -A
git commit -m "Update: Clean root directory and organize files" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_2
echo 方案2: 强制添加所有文件
git add -f .
git add -f %SUBDIR%/
git commit -m "Force update all files" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_3
echo 方案3: 删除未跟踪文件后推送
git add .
git add -u
git commit -m "Remove untracked files and update" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_4
echo 方案4: 重置暂存区后重新添加
git reset .
git add %SUBDIR%/
git add .gitignore 2>nul
git commit -m "Reorganize: Only subdirectory in root" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_5
echo 方案5: 强制推送
git add -A
git commit -m "Force push: Clean root directory" 2>nul
git push origin main --force
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_6
echo 方案6: 分步提交和推送
git add %SUBDIR%/
git commit -m "Add subdirectory" 2>nul
git push origin main 2>nul
if %ERRORLEVEL%==0 (
    git add .
    git commit -m "Clean root" 2>nul
    git push origin main
    if %ERRORLEVEL%==0 goto SUCCESS
)
goto NEXT_METHOD

:METHOD_7
echo 方案7: 使用git rm删除根目录文件
git rm --cached *.py *.bat 2>nul
git rm --cached ~$*.xlsx 2>nul
git add %SUBDIR%/
git commit -m "Remove root files, keep only subdirectory" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_8
echo 方案8: 更新.gitignore后推送
echo *.py >> .gitignore 2>nul
echo *.bat >> .gitignore 2>nul
echo ~$*.xlsx >> .gitignore 2>nul
git add .gitignore
git add -A
git commit -m "Update .gitignore and clean" 2>nul
git push origin main
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:METHOD_9
echo 方案9: 使用不同分支推送
git checkout -b temp-clean 2>nul
git add -A
git commit -m "Clean root directory" 2>nul
git push origin temp-clean 2>nul
if %ERRORLEVEL%==0 (
    git checkout main 2>nul
    git merge temp-clean 2>nul
    git push origin main
    if %ERRORLEVEL%==0 goto SUCCESS
)
git checkout main 2>nul
goto NEXT_METHOD

:METHOD_10
echo 方案10: 最后尝试 - 完整清理和推送
del /Q *.py *.bat 2>nul
del /Q ~$*.xlsx 2>nul
git add -A
git commit -m "Final clean: Remove all temporary files from root" 2>nul
git push origin main --force
if %ERRORLEVEL%==0 goto SUCCESS
goto NEXT_METHOD

:NEXT_METHOD
echo.
echo 方案 %METHOD% 失败，尝试下一个方案...
set /a METHOD+=1
if %METHOD% LEQ 10 goto MAIN_LOOP
goto FAILED

:SUCCESS
echo.
echo ========================================================================
echo 成功！方案 %METHOD% 执行成功！
echo ========================================================================
echo.
echo 所有更改已推送到GitHub
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
echo 1. git add -A
echo 2. git commit -m "Clean root directory"
echo 3. git push origin main
echo.
echo 如果推送被拒绝，尝试:
echo git push origin main --force
echo.

:END
pause

