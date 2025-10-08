
@echo off
setlocal
set EXE_NAME=run.exe
set DIST_DIR=dist
set ZIP_NAME=release.zip

echo Building executable...
pyinstaller --onefile --noconsole run.py
if not exist %DIST_DIR%\%EXE_NAME% (
	echo Build failed! %DIST_DIR%\%EXE_NAME% not found.
	exit /b 1
)

echo Creating release zip...
if exist %ZIP_NAME% del %ZIP_NAME%
powershell -Command "Compress-Archive -Path '%DIST_DIR%\*' -DestinationPath '%ZIP_NAME%'"
if exist %ZIP_NAME% (
	echo Release package created: %ZIP_NAME%
) else (
	echo Failed to create release package!
	exit /b 1
)
endlocal

