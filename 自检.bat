@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   armB model  ::  quick self-check
echo ============================================================
echo.

set "PY="
where python  >nul 2>nul && set "PY=python"
if not defined PY ( where python3 >nul 2>nul && set "PY=python3" )
if not defined PY goto no_python
echo [python]
%PY% -V
echo.

echo [1/3] build the network from config.json, load the weights, run one forward pass
echo       expect: PASS
echo.
%PY% conv_tasnet_spatial.py --check
if errorlevel 1 goto failed
echo.

echo [2/3] spatial feature extractor: ITD / ILD accuracy and shape contract
echo       expect: PASS
echo.
%PY% spatial_feat.py --selftest
if errorlevel 1 goto failed
echo.

set "SAMPLE=%~1"
if not defined SAMPLE for /d %%d in (*) do if not "%%d"=="check_out" if not defined SAMPLE if exist "%%d\*.wav" for %%f in ("%%d\*.wav") do if not defined SAMPLE set "SAMPLE=%%~f"
if not defined SAMPLE goto no_sample
echo [3/3] separate one built-in sample
echo       %SAMPLE%
echo       any sample is fine here - this only proves the whole pipeline runs
echo.
if not exist check_out mkdir check_out
%PY% separate_B.py "%SAMPLE%" --outdir check_out
if errorlevel 1 goto failed

echo.
echo ============================================================
echo   ALL CHECKS PASSED
echo   separated audio is in the  check_out  folder
echo ============================================================
echo.
pause
exit /b 0

:no_python
echo.
echo [ERROR] "python" was not found in PATH.
echo         Install Python 3 first and tick "Add python.exe to PATH".
echo.
pause
exit /b 2

:no_sample
echo.
echo [ERROR] No .wav file found next to this script.
echo         Make sure the sample folder came along, or drag a wav onto
echo         this .bat file to check your own recording.
echo.
pause
exit /b 3

:failed
echo.
echo ============================================================
echo   FAILED - read the error message above.
echo.
echo   missing dependency?   pip install torch numpy soundfile scipy
echo   Windows CPU-only torch:
echo     pip install torch --index-url https://download.pytorch.org/whl/cpu
echo ============================================================
echo.
pause
exit /b 1